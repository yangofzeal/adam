from __future__ import print_function
import math
import time
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hkd_optim import HKDSparseAdam, get_hkd_device, synchronize

DEVICE = get_hkd_device()
DTYPE = torch.float32
DATA = "digits_sparse_hashed_realworld.npz"

STEPS = 20
WARMUP = 5
BATCH = 8
REPEATS = 5
LR = 1e-3
B1, B2 = 0.9, 0.999
EPS = 1e-8
TARGET_UNIONS = [100,250,500,1000,2500,5000,10000,25000,50000]

torch.manual_seed(20260807)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(20260807)

d = np.load(DATA, allow_pickle=False)
base_indices = d["indices"].astype(np.int64)
values_np = d["values"].astype(np.float32)
offsets = d["offsets"].astype(np.int64)
labels = d["labels"].astype(np.int64)
N = int(d["hash_dim"][0])

try:
    from hkd_optim import profile_npz
    DATASET_PROFILE = profile_npz(DATA)
    RESTRICTED = True
except ImportError:
    DATASET_PROFILE = None
    RESTRICTED = False

USE_TRUE_SPARSE = DEVICE.type in ("cuda", "cpu")

def sync():
    synchronize(DEVICE)

def make_remapped_indices(target_union):
    x = base_indices.astype(np.uint64)
    mixed = x * np.uint64(11400714819323198485) + np.uint64(7046029254386353131)
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
        target_t = torch.tensor((labels[sample_ids] >= 5).astype(np.float32), device=DEVICE, dtype=DTYPE)
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
    emb = nn.Embedding(N, 1, sparse=USE_TRUE_SPARSE, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        emb.weight.zero_()
    return emb

class DenseMaskedSparseAdamReference(object):
    def __init__(self, embedding, union):
        self.embedding = embedding
        self.union = union
        self.m = torch.zeros(len(union), device=DEVICE, dtype=DTYPE)
        self.v = torch.zeros(len(union), device=DEVICE, dtype=DTYPE)
        self.step_num = 0

    @torch.no_grad()
    def step(self):
        grad = self.embedding.weight.grad
        union_grad = grad[self.union, 0]
        mask = union_grad != 0
        if not bool(mask.any()):
            return
        pos = torch.nonzero(mask, as_tuple=False).squeeze(1)
        rows = self.union[pos]
        g = union_grad[pos]
        self.step_num += 1
        t = self.step_num
        m_new = self.m[pos]*B1 + g*(1.0-B1)
        v_new = self.v[pos]*B2 + g.square()*(1.0-B2)
        self.m[pos] = m_new
        self.v[pos] = v_new
        step_size = LR*math.sqrt(1.0-B2**t)/(1.0-B1**t)
        self.embedding.weight[rows,0] -= step_size*(m_new/v_new.sqrt().add_(EPS))

def run_ref(batches, union):
    emb = make_embedding()
    if USE_TRUE_SPARSE:
        opt = torch.optim.SparseAdam(emb.parameters(), lr=LR, betas=(B1,B2), eps=EPS)
    else:
        opt = DenseMaskedSparseAdamReference(emb, union)
    times = []
    last_loss = None
    for step, batch in enumerate(batches):
        if USE_TRUE_SPARSE:
            opt.zero_grad(set_to_none=True)
        else:
            emb.weight.grad = None
        loss = forward_loss(emb, batch)
        loss.backward()
        sync()
        t0 = time.perf_counter_ns()
        opt.step()
        sync()
        if step >= WARMUP:
            times.append((time.perf_counter_ns()-t0)/1e6)
        last_loss = float(loss)
    return emb.weight.detach(), times, last_loss

def run_hkd(batches, union):
    emb = make_embedding()
    if RESTRICTED:
        opt = HKDSparseAdam(emb, union, dataset_profile=DATASET_PROFILE, lr=LR, betas=(B1,B2), eps=EPS)
    else:
        opt = HKDSparseAdam(emb, union, lr=LR, betas=(B1,B2), eps=EPS)
    times = []
    last_loss = None
    for step, batch in enumerate(batches):
        emb.weight.grad = None
        loss = forward_loss(emb, batch)
        loss.backward()
        sync()
        t0 = time.perf_counter_ns()
        opt.step()
        sync()
        if step >= WARMUP:
            times.append((time.perf_counter_ns()-t0)/1e6)
        last_loss = float(loss)
    return emb.weight.detach(), times, last_loss


print("REALWORLD_DIGITS_SPARSEADAM_HKD_SWEEP_PORTABLE")
print("LABEL=NON_CHEAT_SPARSE_GRADIENT_SWEEP")
print("SEMANTICS=PYTORCH_SPARSEADAM_MASKED_ADAM")
print("torch={}".format(torch.__version__))
print("device={}".format(DEVICE.type))
if DEVICE.type == "cuda":
    print("accelerator={}".format(torch.cuda.get_device_name(0)))
elif DEVICE.type == "mps":
    print("accelerator=Apple Metal/MPS")
else:
    print("accelerator=CPU")
print("gradient_mode={}".format("sparse_coo" if USE_TRUE_SPARSE else "dense_mps_masked"))
print("restricted={}".format(RESTRICTED))
print("hash_dim={}".format(N))
print("steps={},warmup={},batch={},repeats={}".format(STEPS, WARMUP, BATCH, REPEATS))
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
        ps, stimes, sloss = run_ref(batches, union)
        ph, htimes, hloss = run_hkd(batches, union)

        sparse_all.extend(stimes)
        hkd_all.extend(htimes)

        diff = float((ps[union, 0] - ph[union, 0]).abs().max())
        diffs.append(diff)
        loss_diffs.append(abs(sloss - hloss))

        del ps, ph
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

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
        "{},{},{:.10f},"
        "{:.6f},{:.6f},"
        "{:.6f},{:.6f},"
        "{:.6f},{:.6f},"
        "{:.3e},{:.3e},{}".format(
            target_union, U, U / float(N),
            smean, smed,
            hmean, hmed,
            gain_mean, gain_median,
            maxdiff, ldiff, exact
        )
    )

valid_mean = [g for _, g, _, e in summary if e and g > 0]
valid_med = [g for _, _, g, e in summary if e and g > 0]

if valid_mean:
    gmean = math.exp(sum(math.log(x) for x in valid_mean) / len(valid_mean))
    gmed = math.exp(sum(math.log(x) for x in valid_med) / len(valid_med))
else:
    gmean = float("nan")
    gmed = float("nan")

best = max(summary, key=lambda r: r[1])
worst = min(summary, key=lambda r: r[1])

print()
print("SUMMARY")
print("exact_sizes={}/{}".format(sum(1 for r in summary if r[3]), len(summary)))
print("geomean_speedup_mean={:.6f}x".format(gmean))
print("geomean_speedup_median={:.6f}x".format(gmed))
print("best_actual_union={}".format(best[0]))
print("best_mean_speedup={:.6f}x".format(best[1]))
print("worst_actual_union={}".format(worst[0]))
print("worst_mean_speedup={:.6f}x".format(worst[1]))
print("hkd_faster_all_sizes={}".format(all(r[1] > 1.0 for r in summary)))
