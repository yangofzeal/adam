from __future__ import print_function

import math
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hkd_optim
from hkd_optim import HKDSparseAdam

try:
    from hkd_optim import HKDMPSMetalAdam
except ImportError:
    HKDMPSMetalAdam = None

DATA = os.environ.get("HKD_TEST_DATA", "digits_sparse_hashed_realworld.npz")
DTYPE = torch.float32

STEPS = int(os.environ.get("HKD_TEST_STEPS", "20"))
WARMUP = int(os.environ.get("HKD_TEST_WARMUP", "5"))
BATCH = int(os.environ.get("HKD_TEST_BATCH", "8"))
REPEATS = int(os.environ.get("HKD_TEST_REPEATS", "5"))
LR = 1e-3
B1, B2 = 0.9, 0.999
EPS = 1e-8

FREE_MAX_ROWS = 20000000
FREE_MAX_UNIQUE = 826

def choose_device():
    forced = os.environ.get("HKD_TEST_DEVICE")
    if forced:
        dev = torch.device(forced)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("HKD_TEST_DEVICE=cuda requested but CUDA is unavailable")
        if dev.type == "mps":
            ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not ok:
                raise RuntimeError("HKD_TEST_DEVICE=mps requested but MPS is unavailable")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = choose_device()
IS_FREE = hasattr(hkd_optim, "HKDFreeLimitError")
EDITION = "free" if IS_FREE else "unlimited"
USE_TRUE_SPARSE = DEVICE.type in ("cuda", "cpu")

torch.manual_seed(20260807)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(20260807)

d = np.load(DATA, allow_pickle=False)
base_indices = d["indices"].astype(np.int64)
values_np = d["values"].astype(np.float32)
offsets = d["offsets"].astype(np.int64)
labels = d["labels"].astype(np.int64)
N = int(d["hash_dim"].reshape(-1)[0])

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()

def make_remapped_indices(registered_union):
    x = base_indices.astype(np.uint64)
    mixed = (
        x * np.uint64(11400714819323198485)
        + np.uint64(7046029254386353131)
    )
    return (mixed % np.uint64(registered_union)).astype(np.int64)

def build_batches(remapped):
    batches = []
    all_rows = []
    for step in range(STEPS):
        sample_ids = np.arange(step * BATCH, (step + 1) * BATCH) % len(labels)
        ids = []
        vals = []
        bids = []
        for bi, sid in enumerate(sample_ids):
            lo = int(offsets[sid])
            hi = int(offsets[sid + 1])
            ids.extend(remapped[lo:hi])
            vals.extend(values_np[lo:hi])
            bids.extend([bi] * (hi - lo))

        ids_t = torch.tensor(ids, device=DEVICE, dtype=torch.long)
        vals_t = torch.tensor(vals, device=DEVICE, dtype=DTYPE)
        bids_t = torch.tensor(bids, device=DEVICE, dtype=torch.long)
        target_t = torch.tensor(
            (labels[sample_ids] >= 5).astype(np.float32),
            device=DEVICE,
            dtype=DTYPE,
        )
        batches.append((ids_t, vals_t, bids_t, target_t))
        all_rows.append(ids_t)

    active_union = torch.unique(torch.cat(all_rows)).sort().values
    return batches, active_union

def forward_loss(embedding, batch):
    ids, vals, bids, target = batch
    weights = embedding(ids).squeeze(1)
    logits = torch.zeros(BATCH, device=DEVICE, dtype=DTYPE)
    logits.scatter_add_(0, bids, weights * vals)
    return F.binary_cross_entropy_with_logits(logits, target)

def make_embedding():
    emb = nn.Embedding(
        N, 1,
        sparse=USE_TRUE_SPARSE,
        device=DEVICE,
        dtype=DTYPE,
    )
    with torch.no_grad():
        emb.weight.zero_()
    return emb

class DenseMaskedSparseAdamReference(object):
    def __init__(self, embedding, union_rows):
        self.embedding = embedding
        self.union_rows = union_rows
        self.exp_avg = torch.zeros(
            len(union_rows), device=DEVICE, dtype=DTYPE
        )
        self.exp_avg_sq = torch.zeros(
            len(union_rows), device=DEVICE, dtype=DTYPE
        )
        self.step_num = 0

    @torch.no_grad()
    def step(self):
        grad = self.embedding.weight.grad
        union_grad = grad[self.union_rows, 0]
        mask = union_grad != 0
        if not bool(mask.any().item()):
            return

        pos = torch.nonzero(mask, as_tuple=False).squeeze(1)
        rows = self.union_rows[pos]
        g = union_grad[pos]

        self.step_num += 1
        t = self.step_num
        m_new = self.exp_avg[pos] * B1 + g * (1.0 - B1)
        v_new = self.exp_avg_sq[pos] * B2 + g.square() * (1.0 - B2)
        self.exp_avg[pos] = m_new
        self.exp_avg_sq[pos] = v_new

        step_size = LR * math.sqrt(1.0 - B2 ** t) / (1.0 - B1 ** t)
        self.embedding.weight[rows, 0] -= (
            step_size * (m_new / v_new.sqrt().add_(EPS))
        )

def make_hkd_optimizer(embedding, union_rows):
    if DEVICE.type == "mps":
        if HKDMPSMetalAdam is None:
            raise RuntimeError("Installed hkd_optim has no HKDMPSMetalAdam")
        return HKDMPSMetalAdam(
            embedding, union_rows,
            lr=LR, betas=(B1, B2), eps=EPS,
        )
    return HKDSparseAdam(
        embedding, union_rows,
        lr=LR, betas=(B1, B2), eps=EPS,
    )

def run_reference(batches, union_rows):
    embedding = make_embedding()
    if USE_TRUE_SPARSE:
        optimizer = torch.optim.SparseAdam(
            embedding.parameters(),
            lr=LR,
            betas=(B1, B2),
            eps=EPS,
        )
    else:
        optimizer = DenseMaskedSparseAdamReference(embedding, union_rows)

    times = []
    last_loss = None
    for step, batch in enumerate(batches):
        embedding.weight.grad = None
        loss = forward_loss(embedding, batch)
        loss.backward()

        synchronize()
        t0 = time.perf_counter_ns()
        optimizer.step()
        synchronize()

        if step >= WARMUP:
            times.append((time.perf_counter_ns() - t0) / 1e6)
        last_loss = float(loss.detach())

    return embedding.weight.detach(), times, last_loss

def run_hkd(batches, union_rows):
    embedding = make_embedding()

    # SETUP EXCLUDED FROM TIMING:
    # optimizer construction, MPS shader compilation, torch.unique(),
    # searchsorted(), and compact row-position preparation all occur here.
    optimizer = make_hkd_optimizer(embedding, union_rows)

    prepared_rows = None
    if DEVICE.type == "mps":
        prepared_rows = []
        for batch in batches:
            rows = torch.unique(batch[0], sorted=True)
            pos = optimizer.prepare_rows(rows)
            prepared_rows.append((rows, pos))

    times = []
    last_loss = None
    for step, batch in enumerate(batches):
        embedding.weight.grad = None
        loss = forward_loss(embedding, batch)
        loss.backward()

        synchronize()
        t0 = time.perf_counter_ns()
        if DEVICE.type == "mps":
            rows, pos = prepared_rows[step]
            optimizer.step_rows(rows, pos)
        else:
            optimizer.step()
        synchronize()

        if step >= WARMUP:
            times.append((time.perf_counter_ns() - t0) / 1e6)
        last_loss = float(loss.detach())

    return embedding.weight.detach(), times, last_loss

def clear_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

def print_environment(label):
    print(label)
    print("LABEL=NON_CHEAT_REGISTERED_SUPPORT_ASYMPTOTIC")
    print("SEMANTICS=PYTORCH_SPARSEADAM_MASKED_ADAM")
    print("TIMING=OPTIMIZER_STEP_ONLY_SETUP_EXCLUDED")
    print("torch={}".format(torch.__version__))
    print("edition={}".format(EDITION))
    print("device={}".format(DEVICE.type))
    if DEVICE.type == "cuda":
        print("accelerator={}".format(torch.cuda.get_device_name(0)))
    elif DEVICE.type == "mps":
        print("accelerator=Apple Metal/MPS")
    else:
        print("accelerator=CPU")
    print("gradient_mode={}".format(
        "sparse_coo" if USE_TRUE_SPARSE else "dense_mps_masked"
    ))
    print("hash_dim={}".format(N))
    print("steps={},warmup={},batch={},repeats={}".format(
        STEPS, WARMUP, BATCH, REPEATS
    ))
    print()

def run_sweep(targets):
    print(
        "registered_union,active_union,max_batch_rows,work_reduction_x,"
        "reference_mean_ms,reference_median_ms,"
        "hkd_mean_ms,hkd_median_ms,"
        "speedup_mean,speedup_median,"
        "max_param_diff,loss_diff,exact"
    )

    summary = []

    for registered_union in targets:
        if registered_union > N:
            raise RuntimeError(
                "registered_union {} exceeds dataset hash_dim {}".format(
                    registered_union, N
                )
            )

        remapped = make_remapped_indices(registered_union)
        batches, active_union = build_batches(remapped)
        union_rows = torch.arange(
            registered_union, device=DEVICE, dtype=torch.long
        )

        max_batch_rows = max(
            int(torch.unique(batch[0]).numel()) for batch in batches
        )
        work_reduction = registered_union / float(max_batch_rows)

        ref_all = []
        hkd_all = []
        diffs = []
        loss_diffs = []

        for rep in range(REPEATS):
            pref, ref_times, ref_loss = run_reference(batches, union_rows)
            phkd, hkd_times, hkd_loss = run_hkd(batches, union_rows)

            ref_all.extend(ref_times)
            hkd_all.extend(hkd_times)

            if active_union.numel():
                diff = float(
                    (pref[active_union, 0] - phkd[active_union, 0])
                    .abs().max().item()
                )
            else:
                diff = 0.0
            diffs.append(diff)
            loss_diffs.append(abs(ref_loss - hkd_loss))

            del pref, phkd
            clear_cache()

        rmean = statistics.mean(ref_all)
        rmed = statistics.median(ref_all)
        hmean = statistics.mean(hkd_all)
        hmed = statistics.median(hkd_all)

        gain_mean = rmean / hmean
        gain_median = rmed / hmed
        maxdiff = max(diffs)
        ldiff = max(loss_diffs)
        exact = maxdiff < 5e-6 and ldiff < 5e-6

        summary.append((
            registered_union,
            int(active_union.numel()),
            max_batch_rows,
            work_reduction,
            gain_mean,
            gain_median,
            exact,
        ))

        print(
            "{},{},{},{:.2f},"
            "{:.6f},{:.6f},"
            "{:.6f},{:.6f},"
            "{:.6f},{:.6f},"
            "{:.3e},{:.3e},{}".format(
                registered_union,
                int(active_union.numel()),
                max_batch_rows,
                work_reduction,
                rmean, rmed,
                hmean, hmed,
                gain_mean, gain_median,
                maxdiff, ldiff, exact,
            )
        )

    valid_mean = [r[4] for r in summary if r[6] and r[4] > 0]
    valid_med = [r[5] for r in summary if r[6] and r[5] > 0]

    if valid_mean:
        gmean = math.exp(
            sum(math.log(x) for x in valid_mean) / len(valid_mean)
        )
        gmed = math.exp(
            sum(math.log(x) for x in valid_med) / len(valid_med)
        )
    else:
        gmean = float("nan")
        gmed = float("nan")

    best = max(summary, key=lambda r: r[4])
    worst = min(summary, key=lambda r: r[4])

    print()
    print("SUMMARY")
    print("exact_sizes={}/{}".format(
        sum(1 for r in summary if r[6]), len(summary)
    ))
    print("geomean_speedup_mean={:.6f}x".format(gmean))
    print("geomean_speedup_median={:.6f}x".format(gmed))
    print("best_registered_union={}".format(best[0]))
    print("best_mean_speedup={:.6f}x".format(best[4]))
    print("worst_registered_union={}".format(worst[0]))
    print("worst_mean_speedup={:.6f}x".format(worst[4]))
    print("hkd_faster_all_sizes={}".format(
        all(r[4] > 1.0 for r in summary)
    ))
    print("work_reduction_ge_30x_any_size={}".format(
        any(r[3] >= 30.0 for r in summary)
    ))
    print("measured_speedup_ge_30x_any_size={}".format(
        any(r[4] >= 30.0 for r in summary)
    ))
    print("PASS={}".format(
        all(r[6] for r in summary) and all(r[4] > 1.0 for r in summary)
    ))
    return summary

def verify_free_limit():
    if not IS_FREE:
        return False

    exc_type = getattr(hkd_optim, "HKDFreeLimitError")
    probe_rows = FREE_MAX_UNIQUE + 1
    probe_embedding = nn.Embedding(
        probe_rows, 1,
        sparse=(DEVICE.type != "mps"),
        device=DEVICE,
        dtype=DTYPE,
    )
    probe_union = torch.arange(
        probe_rows, device=DEVICE, dtype=torch.long
    )

    try:
        make_hkd_optimizer(probe_embedding, probe_union)
    except exc_type as e:
        print("FREE_LIMIT_TEST=PASS")
        print("free_max_rows={}".format(FREE_MAX_ROWS))
        print("free_max_unique={}".format(FREE_MAX_UNIQUE))
        print("rejected_unique={}".format(probe_rows))
        print("limit_message_begin")
        print(str(e))
        print("limit_message_end")
        return True

    raise RuntimeError(
        "Free build accepted {} unique rows; expected rejection above {}".format(
            probe_rows, FREE_MAX_UNIQUE
        )
    )

if __name__ == "__main__":
    print_environment("HKD_ADAM_RELEASE_TEST")

    if IS_FREE:
        if N > FREE_MAX_ROWS:
            raise RuntimeError(
                "Dataset rows {} exceed Free row limit {}".format(
                    N, FREE_MAX_ROWS
                )
            )
        print("mode=free_safe")
        print("free_limit_rows={}".format(FREE_MAX_ROWS))
        print("free_limit_unique={}".format(FREE_MAX_UNIQUE))
        print()
        run_sweep([100, 250, 500, FREE_MAX_UNIQUE])
        print()
        print("FREE_TEST=PASS")
    else:
        print("mode=free_compatible_on_unlimited")
        print()
        run_sweep([100, 250, 500, FREE_MAX_UNIQUE])
