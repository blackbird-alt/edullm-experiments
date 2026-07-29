# RUNBOOK — Skill-DAG, GPU phases

For whoever has cluster/GPU access. Phases are ordered by cost and by how much they can
save you: Phase 1 is a few minutes and tells you whether Phases 2 and 3 are even possible.

**Total: ~525 A100-equivalent GPU-hours** (~35 fitting, ~490 main) plus ~93 GB of disk.
Runs are independent and embarrassingly parallel — one process per GPU, no multi-GPU
training, no interconnect requirement. Wall clock is 525 ÷ (GPUs available).

> **STOP — do not start Phase 2 or 3 yet.** 18 review questions are open in
> [PLAN.md](PLAN.md) and 9 values are unset in [PREREG.md](PREREG.md). Item 13 in
> particular can silently null the experiment: the A_ij probe currently holds *total*
> tokens fixed rather than *j's*, which biases every matrix entry negative, and the
> Skill-It `max(A, 0)` clip then freezes the adaptive arms at their starting weights. You
> would get a clean-looking null that means nothing. Phase 0 and Phase 1 are safe to run
> now — they cost almost nothing and are not affected.

## Setup (once)

```bash
module load python 2>/dev/null || true
python3 -m venv ../.venv_skilldag
source ../.venv_skilldag/bin/activate

# Install torch matched to this cluster's CUDA first -- the PyPI default may not
# match the driver. See https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

No model snapshot step is needed: `allenai/OLMo-1B-hf` is public and ungated, and the
revision is pinned in code (`BASE_REVISION = "step1000-tokens4B"`). If your nodes have no
internet, snapshot it on the login node first:

```bash
hf download allenai/OLMo-1B-hf --revision step1000-tokens4B \
  --local-dir $HOME/base/olmo-1b-step1000
```

then pass `--model $HOME/base/olmo-1b-step1000` to every `train_mixture.py` /
`fit_*.py` call.

## Phase 0 — build the token pools (CPU, no GPU)

Free sanity check first, no download:

```bash
python prep_dolma_domains.py --estimate-only
```

Prints the measured natural-weight table across the 9 Dolma domains. If this errors or
returns empty sizes, stop — it means the `Accept-Encoding: identity` path has regressed and
the whole data pipeline is broken.

Then the real thing (~87 GB download, several hours, parallel across shards, resumable):

```bash
sbatch farmshare_phase0_prep.sh
```

Produces `dolma_domains/` — one uint16 `.npy` per domain plus `manifest.json`. Must land on
a real filesystem, not object storage: `DomainPools` memory-maps these at training time.

## Phase 1 — smoke test (GPU, ~10 minutes) ← **do this one first**

```bash
sbatch farmshare_phase1_smoke.sh
```

A single very short run of the real model on real tokens. This is the highest-value step in
the whole runbook because **nothing in this repo has ever touched a GPU** — every check so
far was synthetic data on CPU. It answers: does OLMo-1B load at this revision, does it fit
in memory at batch 8 × accum 8, what is the actual throughput versus the assumed
1.2e14 FLOP/s, and does checkpoint/resume round-trip real weights.

Six of the seven bugs found while building this were only visible when code actually ran.
Expect this to find more.

**Send back:** `runs/smoke/train_log.jsonl`, `runs/smoke/val_log.jsonl`, and the tokens/sec
line from the Slurm output. We use throughput to convert the 525 GPU-h estimate into a real
wall-clock number before anyone commits to Phase 3.

## Phase 2 — fitting runs (GPU, ~35 GPU-h)

**Blocked on the review questions above.** Once they are settled:

```bash
sbatch farmshare_phase2_fit.sh
```

This chains the dependency correctly: the 96-run proxy fleet and the 45-run arm-4 probe
start in parallel; when the fleet finishes, the two CPU fitters and the clusterer run; then
the 10-run arm-5 probe starts.

Both `fit_proxy_fleet.py` and `fit_aij.py` loop over their runs **sequentially** in one
process and append to a JSONL as they go, skipping completed work on restart. So if the job
hits the Slurm time limit, just resubmit — it picks up where it stopped. They do not shard
across nodes; parallelising them would need a `--shard/--shard-index` flag that does not
exist yet.

Sanity check before trusting the output:

```bash
python -c "import json; a=json.load(open('aij_arm4/aij.json'))['A']; \
v=[x for r in a.values() for x in r.values()]; \
print('positive:', sum(1 for x in v if x>0), '/', len(v))"
```

If that prints 0 positive entries, **stop**. Either the probes were too short (review
item 5) or the token-matching confound (item 13) has swallowed the signal. Both produce an
all-zero A, both make arms 4 and 5 meaningless, and the fixes are different.

## Phase 3 — the 15 main runs (GPU, ~490 GPU-h)

**Requires the prereg to be filed first.** Program rule: arms and thresholds are registered
before the real spend.

One decision the launcher needs and the source doc does not settle: the doc says arms 4 and
5 "select optimal domain weight to start, and then adjust the weights 5 times". It does not
say whether "optimal to start" means the natural mix or arm 2's fitted optimum. The launcher
defaults to `natural`; set `ADAPTIVE_INIT=weights_arm2.json` to use the other reading. Pick
one, record it in the prereg, and do not change it after seeing results.

```bash
sbatch farmshare_phase3_main.sh          # Slurm array, 15 independent runs
```

All 15 are resume-safe (`--resume` reloads model, optimizer, scheduler, step, token count,
per-domain cursors, wrap counts, current weights and the reweighting-round boundaries).
Resubmitting the array re-enters any run that was cut off.

## Phase 4 — analysis (CPU)

```bash
python analyze.py --runs "runs/arm*" --margin 0.05 \
  --fitting-costs aij_arm4/aij.json aij_arm5/aij.json fleet/fleet_cost.json \
  --out analysis.json
```

`--margin` is the preregistered non-inferiority threshold. Use the number in PREREG.md; do
not pick it here.

## Do not "fix" these

- **Constant LR, no decay.** Decay would down-weight late-arriving data — in adaptive mode
  that is exactly the data the reweighting chose, which confounds the comparison.
- **Held-out val slices are reserved before any sampling.** No arm trains on them.
- **`Accept-Encoding: identity`** on olmo-data.org requests. The host double-gzips for
  clients advertising gzip, dropping Content-Length and corrupting the stream. `curl` works
  without it only because it omits the header; Python `requests` sends it by default.
- **Gradient checkpointing is off by default on the proxies.** They are ~100M and fit
  trivially; enabling it costs 25–30% throughput across 141 runs for nothing.
- **Fitting compute is logged separately** from training compute and must never be pooled.

## If something breaks

Most likely friction is Phase 1, since nothing here is cluster-tested: OOM at the assumed
batch size, a `transformers` version too old for the OLMo architecture (needs ≥4.40), or
the revision not resolving. Send the traceback plus `nvidia-smi` output — these are usually
one-line fixes.
