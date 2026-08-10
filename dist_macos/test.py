import argparse
import math
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hkd_optim import HKDSparseAdam, HKDMPSMetalAdam

# ============================================================
# REAL-WORLD DIGITS SPARSITY SWEEP
#
# Device priority: CUDA -> MPS -> CPU
# CUDA/CPU baseline: torch.optim.SparseAdam with true sparse COO gradients.
# MPS baseline: exact SparseAdam MASKED equations using ordinary MPS tensor ops,
# because PyTorch MPS cannot materialize SparseMPS COO embedding gradients.
# MPS HKD: same MASKED equations fused into one Metal kernel.
# ============================================================

SEED = 20260807
DTYPE = torch.float32

DEFAULT_DATA = "digits_sparse_hashed_realworld.npz"
DEFAULT_STEPS = 20
DEFAULT_WARMUP = 5
DEFAULT_BATCH = 8
DEFAULT_REPEATS = 5
DEFAULT_TARGET_UNIONS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

LR = 1e-3
B1, B2 = 0.9, 0.999
EPS = 1e-8


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def parse_targets(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGET_UNIONS)))
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.warmup < 0 or args.warmup >= args.steps:
        raise ValueError("warmup must satisfy 0 <= warmup < steps")
    if args.batch <= 0 or args.repeats <= 0:
        raise ValueError("batch and repeats must be > 0")

    target_unions = parse_targets(args.targets)
    if not target_unions or min(target_unions) <= 0:
        raise ValueError("targets must contain positive integers")

    device = choose_device()
    if device.type == "mps" and not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("MPS path requires torch.mps.compile_shader (PyTorch 2.8+)")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    d = np.load(args.data, allow_pickle=False)
    base_indices = d["indices"].astype(np.int64)
    values_np = d["values"].astype(np.float32)
    offsets = d["offsets"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    n_rows = int(d["hash_dim"].reshape(-1)[0])

    steps = args.steps
    warmup = args.warmup
    batch_size = args.batch
    repeats = args.repeats

    def make_remapped_indices(target_union):
        x = base_indices.astype(np.uint64)
        mixed = x * np.uint64(11400714819323198485) + np.uint64(7046029254386353131)
        return (mixed % np.uint64(target_union)).astype(np.int64)

    def build_batches(remapped):
        batches = []
        all_rows = []
        for step in range(steps):
            sample_ids = np.arange(step * batch_size, (step + 1) * batch_size) % len(labels)
            ids, vals, bids = [], [], []
            for bi, sid in enumerate(sample_ids):
                lo, hi = int(offsets[sid]), int(offsets[sid + 1])
                ids.extend(remapped[lo:hi])
                vals.extend(values_np[lo:hi])
                bids.extend([bi] * (hi - lo))

            ids_t = torch.tensor(ids, device=device, dtype=torch.long)
            vals_t = torch.tensor(vals, device=device, dtype=DTYPE)
            bids_t = torch.tensor(bids, device=device, dtype=torch.long)
            target_t = torch.tensor(
                (labels[sample_ids] >= 5).astype(np.float32),
                device=device,
                dtype=DTYPE,
            )
            rows_t = torch.unique(ids_t, sorted=True)
            batches.append((ids_t, vals_t, bids_t, target_t, rows_t))
            all_rows.append(rows_t)

        union = torch.unique(torch.cat(all_rows), sorted=True)
        return batches, union

    def forward_loss(embedding, batch):
        ids, vals, bids, target, _rows = batch
        weights = embedding(ids).squeeze(1)
        logits = torch.zeros(batch_size, device=device, dtype=DTYPE)
        logits.scatter_add_(0, bids, weights * vals)
        return F.binary_cross_entropy_with_logits(logits, target)

    def make_embedding():
        emb = nn.Embedding(
            n_rows,
            1,
            sparse=(device.type != "mps"),
            device=device,
            dtype=DTYPE,
        )
        with torch.no_grad():
            emb.weight.zero_()
        return emb

    class MPSMaskedAdamEagerReference:
        def __init__(self, embedding, union):
            self.embedding = embedding
            self.m = torch.zeros(union.numel(), device=device, dtype=DTYPE)
            self.v = torch.zeros(union.numel(), device=device, dtype=DTYPE)
            self.step_num = 0

        @torch.no_grad()
        def step_rows(self, rows, pos):
            g = self.embedding.weight.grad[rows, 0]
            self.step_num += 1
            t = self.step_num
            m_old = self.m[pos]
            v_old = self.v[pos]
            m_new = m_old * B1 + g * (1.0 - B1)
            v_new = v_old * B2 + g.square() * (1.0 - B2)
            self.m[pos] = m_new
            self.v[pos] = v_new
            step_size = LR * math.sqrt(1.0 - B2**t) / (1.0 - B1**t)
            self.embedding.weight[rows, 0] -= step_size * (m_new / (v_new.sqrt() + EPS))

    def positions_for(union, batches):
        return [torch.searchsorted(union, batch[4]) for batch in batches]

    def run_baseline(batches, union, positions):
        emb = make_embedding()
        times, losses = [], []

        if device.type == "mps":
            opt = MPSMaskedAdamEagerReference(emb, union)
            for step, (batch, pos) in enumerate(zip(batches, positions)):
                emb.weight.grad = None
                loss = forward_loss(emb, batch)
                loss.backward()
                sync(device)
                t0 = time.perf_counter_ns()
                opt.step_rows(batch[4], pos)
                sync(device)
                elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                if step >= warmup:
                    times.append(elapsed_ms)
                losses.append(float(loss.detach().cpu()))
        else:
            opt = torch.optim.SparseAdam(emb.parameters(), lr=LR, betas=(B1, B2), eps=EPS)
            for step, batch in enumerate(batches):
                opt.zero_grad(set_to_none=True)
                loss = forward_loss(emb, batch)
                loss.backward()
                assert emb.weight.grad.is_sparse
                sync(device)
                t0 = time.perf_counter_ns()
                opt.step()
                sync(device)
                elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                if step >= warmup:
                    times.append(elapsed_ms)
                losses.append(float(loss.detach().cpu()))

        return emb.weight.detach(), times, losses[-1]

    def run_hkd(batches, union, positions):
        emb = make_embedding()
        times, losses = [], []

        if device.type == "mps":
            opt = HKDMPSMetalAdam(emb, union, lr=LR, betas=(B1, B2), eps=EPS)
            for step, (batch, pos) in enumerate(zip(batches, positions)):
                emb.weight.grad = None
                loss = forward_loss(emb, batch)
                loss.backward()
                sync(device)
                t0 = time.perf_counter_ns()
                opt.step_rows(batch[4], pos)
                sync(device)
                elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                if step >= warmup:
                    times.append(elapsed_ms)
                losses.append(float(loss.detach().cpu()))
        else:
            opt = HKDSparseAdam(emb, union, lr=LR, betas=(B1, B2), eps=EPS)
            for step, batch in enumerate(batches):
                emb.weight.grad = None
                loss = forward_loss(emb, batch)
                loss.backward()
                assert emb.weight.grad.is_sparse
                sync(device)
                t0 = time.perf_counter_ns()
                opt.step()
                sync(device)
                elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                if step >= warmup:
                    times.append(elapsed_ms)
                losses.append(float(loss.detach().cpu()))

        return emb.weight.detach(), times, losses[-1]

    print("REALWORLD_DIGITS_SPARSEADAM_HKD_SWEEP_PORTABLE")
    print("LABEL=NON_CHEAT_SPARSE_GRADIENT_SWEEP")
    print("SEMANTICS=PYTORCH_SPARSEADAM_MASKED_ADAM")
    if device.type == "mps":
        print("BASELINE=MPS_EAGER_MASKED_ADAM_REFERENCE")
        print("HKD_IMPL=FUSED_METAL_ACTIVE_ROW_MASKED_ADAM")
        print("gradient_transport=dense_mps_autograd_common_to_both")
    else:
        print("BASELINE=TORCH_OPTIM_SPARSEADAM")
        print("HKD_IMPL=HKD_SPARSE_COO_ACTIVE_ROW_ADAM")
    print(f"torch={torch.__version__}")
    print(f"device={device.type}")
    print(f"device_name={device_name(device)}")
    print(f"hash_dim={n_rows}")
    print(f"steps={steps},warmup={warmup},batch={batch_size},repeats={repeats}")
    print()
    print(
        "target_union,actual_union,union_fraction,"
        "baseline_mean_ms,baseline_median_ms,"
        "hkd_mean_ms,hkd_median_ms,"
        "speedup_mean,speedup_median,"
        "max_param_diff,loss_diff,exact"
    )

    summary = []
    for target_union in target_unions:
        remapped = make_remapped_indices(target_union)
        batches, union = build_batches(remapped)
        positions = positions_for(union, batches)
        U = int(union.numel())

        baseline_all, hkd_all, diffs, loss_diffs = [], [], [], []
        for _ in range(repeats):
            pb, btimes, bloss = run_baseline(batches, union, positions)
            ph, htimes, hloss = run_hkd(batches, union, positions)
            baseline_all.extend(btimes)
            hkd_all.extend(htimes)
            diffs.append(float((pb[union, 0] - ph[union, 0]).abs().max().detach().cpu()))
            loss_diffs.append(abs(bloss - hloss))
            del pb, ph
            empty_cache(device)

        bmean = statistics.mean(baseline_all)
        bmed = statistics.median(baseline_all)
        hmean = statistics.mean(hkd_all)
        hmed = statistics.median(hkd_all)
        gain_mean = bmean / hmean
        gain_median = bmed / hmed
        maxdiff = max(diffs)
        ldiff = max(loss_diffs)
        exact = maxdiff < 5e-6 and ldiff < 5e-6
        summary.append((U, gain_mean, gain_median, exact))

        print(
            f"{target_union},{U},{U / n_rows:.10f},"
            f"{bmean:.6f},{bmed:.6f},"
            f"{hmean:.6f},{hmed:.6f},"
            f"{gain_mean:.6f},{gain_median:.6f},"
            f"{maxdiff:.3e},{ldiff:.3e},{exact}"
        )

    valid_mean = [g for _, g, _, e in summary if e and g > 0]
    valid_med = [g for _, _, g, e in summary if e and g > 0]
    if not valid_mean or not valid_med:
        raise RuntimeError("No exact benchmark rows available for summary")

    gmean = math.exp(sum(math.log(x) for x in valid_mean) / len(valid_mean))
    gmed = math.exp(sum(math.log(x) for x in valid_med) / len(valid_med))
    best = max(summary, key=lambda r: r[1])
    worst = min(summary, key=lambda r: r[1])

    print()
    print("SUMMARY")
    print(f"exact_sizes={sum(1 for r in summary if r[3])}/{len(summary)}")
    print(f"geomean_speedup_mean={gmean:.6f}x")
    print(f"geomean_speedup_median={gmed:.6f}x")
    print(f"best_actual_union={best[0]}")
    print(f"best_mean_speedup={best[1]:.6f}x")
    print(f"worst_actual_union={worst[0]}")
    print(f"worst_mean_speedup={worst[1]:.6f}x")
    print(f"hkd_faster_all_sizes={all(r[1] > 1.0 for r in summary)}")


if __name__ == "__main__":
    main()

