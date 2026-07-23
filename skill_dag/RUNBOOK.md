# RUNBOOK — skill-DAG experiment, GPU phase

For whoever has cluster/GPU access. Everything CPU-side is already done and
frozen (dataset, schedules, analysis). Your part is 4 commands now, 12 later.
Any single 24GB+ GPU works for Phase 1; the model is ~0.9B params.

## Setup (once)

```bash
pip install torch transformers datasets
git clone <this-repo>   # or copy the skill_dag/ folder
cd skill_dag
```

## Phase 1 — run these now (~1-2 GPU-hours total)

### 1. Snapshot the base model to shared storage (pinned revision)

```bash
hf download allenai/OLMo-Ladder-760M-0.5xC \
  --revision c8f6875e640a985de0709e8be1f38eef0cb4bc48 \
  --local-dir /SHARED/base/olmo-ladder-760m-05xc
```

The pinned revision matters: this is the last public OLMo ladder checkpoint and
it could be pulled; the local copy becomes our frozen base. Post the /SHARED path
in the channel — three other experiments will branch from it.

### 2. Pilot: cold per-skill accuracy (~15 min)

```bash
python pilot_calibrate.py --model /SHARED/base/olmo-ladder-760m-05xc
```

Prints a per-skill table + writes `pilot_results.json`. FLOOR on the hard skills
is expected cold — don't retune anything yet.

### 3. Smoke run: 200M tokens on one schedule (~30-60 min on one A100)

```bash
python train_cpt.py --model /SHARED/base/olmo-ladder-760m-05xc \
  --schedule schedules/random_201.idx --token-budget 200000000 \
  --ckpt-tokens 25000000 --out runs/smoke_random201 \
  --wandb eduLLM/pretraining
```

(`--wandb` is optional — drop it if W&B isn't set up on the node; local JSONL
logs are always written either way. Runs land in the org's existing `pretraining`
project, grouped as `skill-dag-sequencing`, named `skilldag-<schedule>`.)

### 4. Eval the smoke checkpoints (~15 min)

```bash
python eval_mastery.py --run runs/smoke_random201
```

### Send back (that's all for Phase 1)
- `pilot_results.json`
- `runs/smoke_random201/eval_log.jsonl`
- `runs/smoke_random201/train_log.jsonl`

We use these to (a) confirm skills lift off the floor, (b) set the main-run token
budget, (c) file the preregistration with P8.

**STOP HERE.** Do not launch main runs before the prereg is filed — program rule:
arms and thresholds are registered before the real GPU spend.

## Phase 2 — after we send you the budget number B (~10-40 GPU-hours)

Six independent runs (parallelize freely; different GPUs fine). Identical except
`--schedule`:

```bash
for s in topo_101 topo_102 topo_103 random_201 random_202 random_203; do
  python train_cpt.py --model /SHARED/base/olmo-ladder-760m-05xc \
    --schedule schedules/$s.idx --token-budget $B --out runs/$s
done
```

Then eval each (resume-safe, can run as checkpoints appear):

```bash
for s in topo_101 topo_102 topo_103 random_201 random_202 random_203; do
  python eval_mastery.py --run runs/$s
done
```

Send back the six `runs/*/eval_log.jsonl`. Analysis + verdict runs on our side
(`analyze.py`, preregistered decision rule).

## Do-not-"fix" list (deliberate choices, changing them voids the experiment)
- `train_cpt.py` uses constant LR + warmup, NOT cosine. Decay would down-weight
  late-stream data and confound the order manipulation.
- The dataloader never shuffles. The schedule file IS the experiment.
- All six Phase-2 runs must share the same base path, budget, and hyperparameters.
- bf16 cast of the F32 weights is intended.

## If something breaks
Padding/generation quirks in pilot/eval are the most likely friction (these
scripts are cluster-untested). Send the traceback — fixes are usually one line.
