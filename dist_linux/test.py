from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE / "dist"))

from hkd_optim.sparse_adam import HKDSparseAdam

import math, time, statistics, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# REAL-WORLD DIGITS SPARSITY SWEEP:
# torch.optim.SparseAdam vs HKD∞
#
# Run:
#   python realworld_digits_sparseadam_hkd_sweep_gpu.py
#
# Requires:
#   digits_sparse_hashed_realworld.npz in same directory
#
# This uses true sparse COO gradients from nn.Embedding(..., sparse=True).
# Both optimizers implement SparseAdam MASKED semantics.
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU required"

torch.manual_seed(20260807)
torch.cuda.manual_seed_all(20260807)

DEVICE = torch.device("cuda")
DTYPE = torch.float32

DATA = "digits_sparse_hashed_realworld.npz"

STEPS = 20
WARMUP = 5
BATCH = 8
REPEATS = 5

LR = 1e-3
B1, B2 = 0.9, 0.999
EPS = 1e-8

# Sweep target active universe sizes.
# These are the approximate numbers of embedding rows available to the
# real digit features during each run.
TARGET_UNIONS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

d = np.load(DATA)
base_indices = d["indices"].astype(np.int64)
values_np = d["values"].astype(np.float32)
offsets = d["offsets"].astype(np.int64)
labels = d["labels"].astype(np.int64)

# Keep the same large sparse parameter bank used in prior T4 experiment.
N = int(d["hash_dim"][0])

def sync():
    torch.cuda.synchronize()

def make_remapped_indices(target_union):
    """
    Deterministically remap the real digit sparse features into a controlled
    active row universe [0, target_union).

    The source examples and values remain the real handwritten digits.
    Only their embedding-row IDs are remapped to control sparsity.
    """
    # Mix the original hashed feature IDs, then constrain to target universe.
    x = base_indices.astype(np.uint64)
    mixed = (x * np.uint64(11400714819323198485) + np.uint64(7046029254386353131))
    return (mixed % np.uint64(target_union)).astype(np.int64)

def build_batches(remapped):
    batches = []
    all_rows = []

    for step in range(STEPS):
        sample_ids = np.arange(step*BATCH, (step+1)*BATCH) % len(labels)

        ids = []
        vals = []
        bids = []

        for bi, sid in enumerate(sample_ids):
            lo, hi = int(offsets[sid]), int(offsets[sid+1])
            ids.extend(remapped[lo:hi])
            vals.extend(values_np[lo:hi])
            bids.extend([bi] * (hi-lo))

        ids_t = torch.tensor(ids, device=DEVICE, dtype=torch.long)
        vals_t = torch.tensor(vals, device=DEVICE, dtype=DTYPE)
        bids_t = torch.tensor(bids, device=DEVICE, dtype=torch.long)
        target_t = torch.tensor(
            (labels[sample_ids] >= 5).astype(np.float32),
            device=DEVICE, dtype=DTYPE
        )

        batches.append((ids_t, vals_t, bids_t, target_t))
        all_rows.append(ids_t)

    union = torch.unique(torch.cat(all_rows)).sort().values
    return batches, union

def forward_loss(embedding, batch):
    ids, vals, bids, target = batch
    weights = embedding(ids).squeeze(1)
    logits = torch.zeros(BATCH, device=DEVICE, dtype=DTYPE)
    logits.scatter_add_(0, bids, weights * vals)
    return F.binary_cross_entropy_with_logits(logits, target)

def make_embedding():
    emb = nn.Embedding(N, 1, sparse=True, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        emb.weight.zero_()
    return emb

def run_sparseadam(batches):
    emb = make_embedding()
    opt = torch.optim.SparseAdam(
        emb.parameters(), lr=LR, betas=(B1,B2), eps=EPS
    )

    times = []
    losses = []

    for step, batch in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss = forward_loss(emb, batch)
        loss.backward()
        assert emb.weight.grad.is_sparse

        sync()
        t0 = time.perf_counter_ns()
        opt.step()
        sync()

        if step >= WARMUP:
            times.append((time.perf_counter_ns()-t0)/1e6)
        losses.append(float(loss))

    return emb.weight.detach(), times, losses[-1]

def run_hkd(batches, union):
    emb = make_embedding()
    opt = HKDSparseAdam(emb, union)

    times = []
    losses = []

    for step, batch in enumerate(batches):
        emb.weight.grad = None
        loss = forward_loss(emb, batch)
        loss.backward()
        assert emb.weight.grad.is_sparse

        sync()
        t0 = time.perf_counter_ns()
        opt.step()
        sync()

        if step >= WARMUP:
            times.append((time.perf_counter_ns()-t0)/1e6)
        losses.append(float(loss))

    return emb.weight.detach(), times, losses[-1]

def pctile(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))

print("REALWORLD_DIGITS_SPARSEADAM_HKD_SWEEP_GPU")
print("LABEL=NON_CHEAT_SPARSE_GRADIENT_SWEEP")
print("SEMANTICS=PYTORCH_SPARSEADAM_MASKED_ADAM")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"hash_dim={N}")
print(f"steps={STEPS},warmup={WARMUP},batch={BATCH},repeats={REPEATS}")
print()
print(
    "target_union,actual_union,union_fraction,"
    "sparseadam_mean_ms,sparseadam_median_ms,"
    "hkd_mean_ms,hkd_median_ms,"
    "speedup_mean,speedup_median,"
    "max_param_diff,loss_diff,exact"
)

summary = []

for target_union in TARGET_UNIONS:
    remapped = make_remapped_indices(target_union)
    batches, union = build_batches(remapped)
    U = len(union)

    sparse_all = []
    hkd_all = []
    diffs = []
    loss_diffs = []

    for rep in range(REPEATS):
        ps, stimes, sloss = run_sparseadam(batches)
        ph, htimes, hloss = run_hkd(batches, union)

        sparse_all.extend(stimes)
        hkd_all.extend(htimes)

        diff = float((ps[union,0]-ph[union,0]).abs().max())
        diffs.append(diff)
        loss_diffs.append(abs(sloss-hloss))

        # Release giant embeddings before next repeat.
        del ps, ph
        torch.cuda.empty_cache()

    smean = statistics.mean(sparse_all)
    smed = statistics.median(sparse_all)
    hmean = statistics.mean(hkd_all)
    hmed = statistics.median(hkd_all)

    gain_mean = smean/hmean
    gain_median = smed/hmed

    maxdiff = max(diffs)
    ldiff = max(loss_diffs)
    exact = maxdiff < 5e-6 and ldiff < 5e-6

    summary.append((U,gain_mean,gain_median,exact))

    print(
        f"{target_union},{U},{U/N:.10f},"
        f"{smean:.6f},{smed:.6f},"
        f"{hmean:.6f},{hmed:.6f},"
        f"{gain_mean:.6f},{gain_median:.6f},"
        f"{maxdiff:.3e},{ldiff:.3e},{exact}"
    )

# Geometric mean is appropriate for multiplicative speed ratios.
valid_mean = [g for _,g,_,e in summary if e and g>0]
valid_med = [g for _,_,g,e in summary if e and g>0]

gmean = math.exp(sum(math.log(x) for x in valid_mean)/len(valid_mean))
gmed = math.exp(sum(math.log(x) for x in valid_med)/len(valid_med))

best = max(summary, key=lambda r:r[1])
worst = min(summary, key=lambda r:r[1])

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
