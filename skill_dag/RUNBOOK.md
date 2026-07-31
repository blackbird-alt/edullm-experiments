# RUNBOOK — Skill-DAG, GPU phases (MIT ORCD / Engaging)

For whoever has cluster access. Phases are ordered by cost and by how much they can save
you: Phase 1 is under an hour and tells you whether Phases 2 and 3 are even possible.

Written for MIT ORCD's [Engaging cluster](https://orcd-docs.mit.edu/), assuming the group
partition with 4 H100s. Runs are independent and embarrassingly parallel — one process per
GPU, no multi-GPU training, no interconnect requirement — so the only things that matter
are how many GPUs you hold at once and how long a single job may live.

> **STOP — do not start Phase 2 or 3 yet.** [DECISIONS.md](DECISIONS.md) lists what has to
> be settled first and what it costs. 18 review questions are open in
> [PLAN.md](PLAN.md) and 9 values are unset in [PREREG.md](PREREG.md). Item 13 in
> particular can silently null the experiment: the A_ij probe currently holds *total*
> tokens fixed rather than *j's*, which biases every matrix entry negative, and the
> Skill-It `max(A, 0)` clip then freezes the adaptive arms at their starting weights. You
> would get a clean-looking null that means nothing. Phase 0 and Phase 1 are safe to run
> now — they cost almost nothing and are not affected.

## What this costs

Per main run: 2B tokens on a 1.18B-parameter model, so ~27 GPU-hours on an H100.

| phase | cost | wall clock on 4 H100s |
|---|---|---|
| 0 — build token pools | CPU, ~93 GB | several hours |
| 1 — smoke test | 1 GPU | under an hour |
| 2 — fitting runs (141 probes) | ~18 GPU-h | ~1.5 hours |
| 3 — 15 main runs | ~400 GPU-h | **~4.3 days** |
| 4/5 — analysis, benchmarks | CPU + inference | ~an hour |

Roughly **4.5 days end to end**, or 3.5 with gradient checkpointing off (see "Making it
cheaper" — it is on unconditionally and 80 GB is almost certainly enough without it).

Two caveats. These assume 40% of dense bf16 peak; **they are estimates until Phase 1
measures the real number**, which is what Phase 1 is for. And gradient checkpointing means
the hardware does ~8ND per token rather than 6ND, which the older 490 GPU-h figure in
PLAN.md did not include.

Reproduce with `python timing_estimate.py`, which also prints the public-partition
fallbacks if the group nodes are tied up.

## Partitions

| partition | max time | GPUs | used for |
|---|---|---|---|
| your group partition | typically 7–14 days | the 4 H100s | Phases 1, 2, 3, 5 |
| `mit_normal` | 12 h | — | Phases 0, 2-fitters, 4 (CPU) |

A whole 27-hour run fits inside one job on the group partition, so there is no chunking,
no preemption and no queue. Phase 3 still supports chunking — it checkpoints and resubmits
itself when a run cannot finish in one job — but on owned nodes it should never need to.

You need three things from `scontrol show partition <name>`: the partition name, its real
time limit, and how many of the 4 H100s are yours rather than shared. Then:

```bash
PARTITION=pi_yourgroup WALL=7-00:00:00 CONC=4 bash orcd_phase3_main.sh
```

`CONC` is how many runs go at once; set it to the GPUs you actually get. Off the public
partitions the launcher requests `--gres=gpu:1` rather than `-G h100:1`, because group
partitions usually do not advertise GPU types — override with `GPU_REQ=` if yours does.

If the group nodes are unavailable, the public fallback is `mit_preemptable` (48 h, 4 GPUs,
jobs killed at any time), which is the launcher default and takes about 4 days on H100s or
13 on the L40S that ORCD hands out by default. `mit_normal_gpu` caps jobs at 6 hours with
2 GPUs, which means ~210 separate queue waits; avoid it for Phase 3.

## Setup (once)

Put the venv in pool storage, not home: home is 200 GB and you will want it for the
checkpoints. Pool is 1 TB and survives between jobs (scratch does too, but is purged
after six months idle).

```bash
module load miniforge          # or: module load python/3.11
python3 -m venv $HOME/orcd/pool/venv_skilldag
source $HOME/orcd/pool/venv_skilldag/bin/activate

# Install torch matched to the cluster's CUDA first -- the PyPI default may not
# match the driver. Check with `nvidia-smi` on a GPU node.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The launchers all source `$HOME/orcd/pool/venv_skilldag/bin/activate`. If you put the venv
elsewhere, edit that one line in each.

### Where the data goes

| what | size | where | why |
|---|---|---|---|
| token pools | ~93 GB | `$HOME/orcd/scratch/skilldag/dolma_domains` | flash, 1 TB, memory-mapped during training |
| run outputs + checkpoints | ~10 GB × 15 | `$HOME/orcd/scratch/skilldag/runs` | resume state, rewritten constantly |
| code, venv, results | small | `$HOME` / pool | home is backed up; scratch and pool are not |

Both default paths are overridable with `DATA=` and `RUNS=`. Neither scratch nor pool is
backed up — copy `analysis.json`, `bench_summary.json` and the `*_log.jsonl` files into
home when a phase finishes. Check quota with `cat ~/orcd/.quota`.

No model snapshot step is needed if compute nodes can reach the internet: `allenai/OLMo-1B-hf`
is public and ungated, and the revision is pinned in code
(`BASE_REVISION = "step1000-tokens4B"`). Engaging compute nodes generally can, but if yours
cannot, snapshot it on a login node first:

```bash
hf download allenai/OLMo-1B-hf --revision step1000-tokens4B \
  --local-dir $HOME/base/olmo-1b-step1000
```

then pass `--model $HOME/base/olmo-1b-step1000` to every `train_mixture.py` / `fit_*.py`
call, and `--base $HOME/base/olmo-1b-step1000` to `eval_benchmarks.py` (it takes `--base`
rather than `--model`, since it also uses the path as the tokenizer source).

If `hf` is not found, your `huggingface_hub` predates the renamed CLI — either upgrade it
or use `huggingface-cli download` with the same arguments.

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
sbatch orcd_phase0_prep.sh
```

Produces `$HOME/orcd/scratch/skilldag/dolma_domains/` — one uint16 `.npy` per domain plus
`manifest.json`. Must land on a real filesystem, not object storage: `DomainPools`
memory-maps these at training time.

`mit_normal` caps jobs at 12 h, which may not cover the download plus tokenization on the
first pass. Completed shards are skipped on restart, so just resubmit until it finishes.

Then export the path for every later phase (or add it to your `.bashrc`):

```bash
export DATA=$HOME/orcd/scratch/skilldag/dolma_domains
```

## Phase 1 — smoke test (GPU, under an hour) ← **do this one first**

```bash
sbatch -p pi_yourgroup orcd_phase1_smoke.sh
```

A short run of the real model on real tokens. This is the highest-value step in the whole
runbook because **nothing in this repo has ever touched a GPU** — every check so far was
synthetic data on CPU. It answers: does OLMo-1B load at this revision, does it fit at
batch 8 × accum 8, what is the actual throughput, and does checkpoint/resume round-trip
real weights.

Six of the seven bugs found while building this were only visible when code actually ran.
Expect this to find more.

**Send back:** `phase1_out/` and the `MEASURED:` line from the Slurm output. Every
wall-clock number above is derived from an assumed fraction of peak FLOPs; that one
measurement replaces all of them. Use `-p` to run it on the same partition and GPU type
Phase 3 will use, or the number does not transfer.

## Phase 2 — fitting runs (GPU, ~18 GPU-h)

**Blocked on the review questions above.** Once they are settled:

```bash
PARTITION=pi_yourgroup WALL=7-00:00:00 NSHARDS=4 \
  I_HAVE_SETTLED_ITEM_13=yes bash orcd_phase2_fit.sh
```

Note `bash`, not `sbatch` — this one is a submitter you run on the login node, and it
refuses to do anything until item 13 is acknowledged. It chains the dependencies: the
96-run proxy fleet and the 45-run arm-4 probe start in parallel, each as a Slurm array;
when the fleet is merged, the two CPU fitters and the clusterer run; then the 10-run arm-5
probe starts.

Each probe is a 50–100M proxy on 200M tokens, about 10 minutes on an H100, so the whole
phase is roughly 1.5 hours across 4 GPUs. Match `NSHARDS` to the GPUs you own; above that
the extra tasks just queue. Arm 5 has only 10 probes and is capped at 10 tasks however
high you set this. `GPU_REQ` and `THROTTLE` (e.g. `THROTTLE=%2`) are the other knobs.
The CPU fitter and merge jobs always go to `mit_normal`, which anyone can use.

Each array task runs every Nth probe and appends to its own `*.shardNN.jsonl`; a short CPU
job afterwards merges those into the canonical `fleet.jsonl` / `probes.jsonl` and writes
`aij.json`. The merge is a separate job on purpose — on a shared filesystem the
last-finishing task cannot be relied on to see its siblings' final writes.

Everything is resumable at run granularity, so a task killed by the time limit loses at
most the one probe it was in the middle of. Resubmit the same command and completed runs
are skipped. If a merge job reports outstanding runs, refill the gaps by resubmitting the
array, then rerun the merge alone with `--assemble-only`.

Sanity check before trusting the output:

```bash
python -c "import json; a=json.load(open('aij_arm4/aij.json'))['A']; \
v=[x for r in a.values() for x in r.values()]; \
print('positive:', sum(1 for x in v if x>0), '/', len(v))"
```

If that prints 0 positive entries, **stop**. Either the probes were too short (review
item 5) or the token-matching confound (item 13) has swallowed the signal. Both produce an
all-zero A, both make arms 4 and 5 meaningless, and the fixes are different.

## Phase 3 — the 15 main runs (GPU, 400–1300 GPU-h by GPU type)

**Requires the prereg to be filed first.** Program rule: arms and thresholds are registered
before the real spend.

One decision the launcher needs and the source doc does not settle: the doc says arms 4 and
5 "select optimal domain weight to start, and then adjust the weights 5 times". It does not
say whether "optimal to start" means the natural mix or arm 2's fitted optimum. The launcher
defaults to `natural`; set `ADAPTIVE_INIT=weights_arm2.json` to use the other reading. Pick
one, record it in the prereg, and do not change it after seeing results.

```bash
PARTITION=pi_yourgroup WALL=7-00:00:00 CONC=4 bash orcd_phase3_main.sh
```

`bash`, not `sbatch`: the script submits itself as a 15-task array and then **each task
resubmits itself until its own run finishes**. No main run fits in a single job window on
any public partition, so each chunk trains up to `--max-seconds`, checkpoints, and exits.
`train_mixture.py` writes `run_config.json` only on completion, which is how a task knows
whether it is done; `MAX_CHUNKS` (default 40) stops a runaway loop.

Preemption is handled on two paths that complement each other. `--requeue` lets Slurm
restart a task that was killed, and it resumes from the last checkpoint. `--signal=USR1@180`
gives the trainer three minutes' warning, and it checkpoints at the next step boundary
rather than mid-optimizer-update. Worst case you lose one checkpoint interval, which
defaults to 30 minutes of wall clock (`--ckpt-every-seconds`).

Progress across all 15:

```bash
bash orcd_phase3_main.sh --status
```

Resume is complete state, not just weights: model, optimizer, scheduler, step, token count,
per-domain read cursors, wrap counts, current weights, and the reweighting-round
boundaries. Killing and resubmitting is safe at any point.

## Phase 4 — analysis (CPU)

```bash
python analyze.py --runs "$RUNS/arm*" --margin 0.05 \
  --fitting-costs aij_arm4/aij.json aij_arm5/aij.json fleet/fleet_cost.json \
  --out analysis.json
```

`--margin` is the preregistered non-inferiority threshold. Use the number in PREREG.md; do
not pick it here.

## Phase 5 — benchmark pass (GPU, inference only, optional)

Secondary and descriptive. The preregistered verdict comes from Phase 4 and nothing here
feeds it. It exists because held-out loss on the nine training domains cannot address the
source doc's claim about benchmark scores (review item 16), and because at inference only
it costs a rounding error against the training spend.

Datasets first, on a login node, so the GPU job needs no network:

```bash
python eval_benchmarks.py --download-only
```

Then on a GPU node — this fits inside `mit_normal_gpu`'s 6 h window:

```bash
python eval_benchmarks.py --runs "$RUNS/arm*" --base-ref --out bench_summary.json
```

`--base-ref` also scores the untrained base revision, which is the reference that tells you
whether 2B tokens moved these benchmarks at all. Each run gets its own `bench.json` and is
skipped on re-runs unless you pass `--force`. Use `--limit 200` for a shakedown; those
numbers are not results.

Expect accuracy at or near chance. That is the honest expected outcome for 1B parameters
and 2B tokens, which is why `acc_per_char` and the continuous `correct_prob_per_char` are
reported next to each task's chance rate. **Do not read a two-point accuracy gap as a
finding.** If the continuous metrics also fail to separate arms, the conclusion is that
this scale cannot resolve benchmark differences — worth reporting, and not a licence to
switch DV after the fact.

## Making it cheaper

In descending order of how much they buy and ascending order of how much they cost you
scientifically. The first is free; the rest change the experiment and belong in the prereg
*before* launch, not after seeing a number you dislike.

1. **Turn off gradient checkpointing** — saves ~25%, changes nothing scientific, takes
   deleting one line in `train_mixture.py`. It is only needed if activations do not fit,
   and a 1B model at batch 8 × 2048 needs roughly 25–35 GB against the H100's 80. Phase 1
   confirms it. Takes the main phase from ~4.3 days to ~3.3. Pull this lever first.
2. **Drop to 2 seeds** — saves a third (10 runs instead of 15). Weakens the variance
   estimate that the non-inferiority test depends on; see PLAN.md item 5 on why 3 was
   chosen.
3. **Halve the token budget to 1B** — saves half. This is the most damaging option. At 2B
   tokens on 1.18B parameters the run is already at a ~2:1 token-to-parameter ratio,
   far under Chinchilla-optimal, and mixture effects are smaller and noisier the shorter
   the run. Halving again risks a null that says nothing about the hypothesis.

Anything that changes tokens, seeds, or arms must be settled before Phase 3 starts.

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

ORCD-specific things that bite:

- **Job rejected for the time limit.** `mit_normal_gpu` allows 6 h, not more. The launchers
  set this per partition; if you override `WALL` by hand, keep it under the cap.
- **Requesting too many CPUs silently costs a GPU.** On the public partitions Engaging
  reserves a fixed number of CPUs per GPU (16 for L40S, 15 for H200); ask for more and the
  job may be allocated an extra GPU against your limit. The launchers request 16.
- **Array stuck pending.** You are at the concurrent-GPU limit (2 or 4). Expected — the
  chain drains it over time. `squeue -u $USER --start` estimates when.
- **Rocky 8 versus CentOS 7.** Modules built on the older nodes will not work on
  `mit_normal*` or `mit_preemptable`. Build the venv on the same OS you run on.
- **Everything vanished from scratch.** Scratch is purged after six months idle and is not
  backed up. Copy results to home as each phase finishes.
