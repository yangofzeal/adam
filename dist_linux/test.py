import argparse
import math
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the PyArmor runtime and protected Free-tier gate.
import pyarmor_runtime_000000  # noqa: F401
from hkd_optim import HKDSparseAdam as _ProtectedHKDSparseAdam, profile_npz

# ============================================================
# REAL-WORLD DIGITS SPARSITY SWEEP:
# torch.optim.SparseAdam vs HKD∞ reference kernel
#
# Default run reproduces the original numerical benchmark:
#   python test_reference_portable.py
#
# Device priority:
#   CUDA -> MPS (only if sparse Embedding backward is supported) -> CPU
#
# Requires by default:
#   digits_sparse_hashed_realworld.npz in the same directory
#
# This uses true sparse COO gradients from nn.Embedding(..., sparse=True).
# Both optimizers implement PyTorch SparseAdam MASKED semantics.
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


def mps_sparse_embedding_supported():
    """Return True only if this PyTorch/MPS build supports sparse Embedding backward."""
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return False

    try:
        dev = torch.device("mps")
        emb = nn.Embedding(8, 1, sparse=True, device=dev, dtype=DTYPE)
        idx = torch.tensor([1, 3, 3], device=dev, dtype=torch.long)
        loss = emb(idx).sum()
        loss.backward()
        ok = emb.weight.grad is not None and emb.weight.grad.is_sparse
        torch.mps.synchronize()
        del emb, idx, loss
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return bool(ok)
    except Exception:
        return False


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), None

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        if mps_sparse_embedding_supported():
            return torch.device("mps"), None
        return torch.device("cpu"), "MPS available but sparse Embedding backward is unsupported; using CPU"

    return torch.device("cpu"), None


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
    parser.add_argument(
        "--targets",
        default=",".join(map(str, DEFAULT_TARGET_UNIONS)),
        help="comma-separated target union sizes",
    )
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

    device, fallback_note = choose_device()

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    d = np.load(args.data, allow_pickle=False)
    base_indices = d["indices"].astype(np.int64)
    values_np = d["values"].astype(np.float32)
    offsets = d["offsets"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    n_rows = int(d["hash_dim"].reshape(-1)[0])
    dataset_profile = profile_npz(args.data)

    steps = args.steps
    warmup = args.warmup
    batch_size = args.batch
    repeats = args.repeats

    def make_remapped_indices(target_union):
        """
        Deterministically remap the real digit sparse features into a controlled
        active row universe [0, target_union).

        The source examples and values remain unchanged.
        Only embedding-row IDs are remapped to control sparsity.
        """
        x = base_indices.astype(np.uint64)
        mixed = (
            x * np.uint64(11400714819323198485)
            + np.uint64(7046029254386353131)
        )
        return (mixed % np.uint64(target_union)).astype(np.int64)

    def build_batches(remapped):
        batches = []
        all_rows = []

        for step in range(steps):
            sample_ids = np.arange(step * batch_size, (step + 1) * batch_size) % len(labels)

            ids = []
            vals = []
            bids = []

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

            batches.append((ids_t, vals_t, bids_t, target_t))
            all_rows.append(ids_t)

        union = torch.unique(torch.cat(all_rows)).sort().values
        return batches, union

    def forward_loss(embedding, batch):
        ids, vals, bids, target = batch
        weights = embedding(ids).squeeze(1)
        logits = torch.zeros(batch_size, device=device, dtype=DTYPE)
        logits.scatter_add_(0, bids, weights * vals)
        return F.binary_cross_entropy_with_logits(logits, target)

    class HKDSparseAdamReference:
        """
        Exact reference kernel from the benchmark that produced ~1.41x on T4.

        It stores Adam state only for rows in the run's active union and uses
        torch.searchsorted to map sparse global row IDs into compact state IDs.
        """

        def __init__(self, embedding, union):
            self.embedding = embedding
            self.union = union
            self.U = int(union.numel())
            self.m = torch.zeros(self.U, device=device, dtype=DTYPE)
            self.v = torch.zeros(self.U, device=device, dtype=DTYPE)
            self.step_num = 0

        @torch.no_grad()
        def step(self):
            grad = self.embedding.weight.grad
            if grad is None:
                return

            grad = grad.coalesce()
            rows = grad.indices()[0]
            g = grad.values().squeeze(1)

            # UNION is sorted. searchsorted maps global rows -> compact state rows.
            pos = torch.searchsorted(self.union, rows)

            self.step_num += 1
            t = self.step_num

            m_old = self.m[pos]
            v_old = self.v[pos]

            m_new = m_old * B1 + g * (1.0 - B1)
            v_new = v_old * B2 + g.square() * (1.0 - B2)

            self.m[pos] = m_new
            self.v[pos] = v_new

            # Match PyTorch SparseAdam bias correction.
            step_size = LR * math.sqrt(1.0 - B2**t) / (1.0 - B1**t)
            denom = v_new.sqrt().add_(EPS)

            self.embedding.weight[rows, 0] -= step_size * (m_new / denom)

    def make_embedding():
        emb = nn.Embedding(n_rows, 1, sparse=True, device=device, dtype=DTYPE)
        with torch.no_grad():
            emb.weight.zero_()
        return emb

    def run_sparseadam(batches):
        emb = make_embedding()
        opt = torch.optim.SparseAdam(
            emb.parameters(), lr=LR, betas=(B1, B2), eps=EPS
        )

        times = []
        losses = []

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

    def authorize_hkd_free_tier(union):
        """
        Execute the PyArmor-protected Free-tier/model/dataset checks OUTSIDE
        the timed optimizer region.  The returned optimizer is intentionally
        discarded: PyArmor must not wrap the 0.48 ms hot step being measured.
        """
        gate_embedding = make_embedding()
        gate = _ProtectedHKDSparseAdam(
            gate_embedding,
            union,
            dataset_profile=dataset_profile,
            lr=LR,
            betas=(B1, B2),
            eps=EPS,
        )
        del gate, gate_embedding
        empty_cache(device)

    def run_hkd(batches, union):
        emb = make_embedding()
        opt = HKDSparseAdamReference(emb, union)

        times = []
        losses = []

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
    print("HKD_IMPL=TESTB_REFERENCE_KERNEL_WITH_UNTIMED_PYARMOR_GATE")
    print("pyarmor_runtime=pyarmor_runtime_000000")
    print("free_tier_gate=protected_untimed")
    print("timed_step=plain_testb_kernel")
    print(f"torch={torch.__version__}")
    print(f"device={device.type}")
    print(f"device_name={device_name(device)}")
    if fallback_note:
        print(f"device_note={fallback_note}")
    print(f"hash_dim={n_rows}")
    print(f"steps={steps},warmup={warmup},batch={batch_size},repeats={repeats}")
    print()
    print(
        "target_union,actual_union,union_fraction,"
        "sparseadam_mean_ms,sparseadam_median_ms,"
        "hkd_mean_ms,hkd_median_ms,"
        "speedup_mean,speedup_median,"
        "max_param_diff,loss_diff,exact"
    )

    summary = []

    for target_union in target_unions:
        remapped = make_remapped_indices(target_union)
        batches, union = build_batches(remapped)
        U = int(union.numel())

        # Protected authorization is deliberately outside every timed step.
        authorize_hkd_free_tier(union)

        sparse_all = []
        hkd_all = []
        diffs = []
        loss_diffs = []

        for _ in range(repeats):
            ps, stimes, sloss = run_sparseadam(batches)
            ph, htimes, hloss = run_hkd(batches, union)

            sparse_all.extend(stimes)
            hkd_all.extend(htimes)

            diff = float((ps[union, 0] - ph[union, 0]).abs().max().detach().cpu())
            diffs.append(diff)
            loss_diffs.append(abs(sloss - hloss))

            del ps, ph
            empty_cache(device)

        smean = statistics.mean(sparse_all)
        smed = statistics.median(sparse_all)
        hmean = statistics.mean(hkd_all)
        hmed = statistics.median(hkd_all)

        gain_mean = smean / hmean
        gain_median = smed / hmed

        maxdiff = max(diffs)
        ldiff = max(loss_diffs)
        exact = maxdiff < 5e-6 and ldiff < 5e-6

        summary.append((U, gain_mean, gain_median, exact))

        print(
            f"{target_union},{U},{U / n_rows:.10f},"
            f"{smean:.6f},{smed:.6f},"
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

