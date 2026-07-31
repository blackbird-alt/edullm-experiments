#!/bin/bash
# Phase 1: FIRST CONTACT WITH A GPU. ~30 minutes, one GPU.
#   sbatch -p pi_yourgroup orcd_phase1_smoke.sh    # run it on the hardware Phase 3 will use
#   sbatch orcd_phase1_smoke.sh                    # public fallback
#
# Command-line flags override the #SBATCH directives below, so -p is how you point this
# at the group partition. Run it on the SAME GPU type as Phase 3 -- the whole purpose is
# the throughput number, and an L40S and an H100 differ by about 3x.
#
# Nothing in this repo has ever run on a GPU or read a real token -- every check so
# far was synthetic data on CPU. This run answers the questions that cannot be
# answered any other way:
#   - does OLMo-1B load at revision step1000-tokens4B?
#   - does it fit at batch 8 x accum 8, WITH and WITHOUT gradient checkpointing?
#   - what is the ACTUAL throughput, which sets the whole wall-clock estimate?
#   - does checkpoint/resume round-trip real model weights?
#
# THE THROUGHPUT NUMBER IS THE POINT. Every cost figure in RUNBOOK.md is derived from an
# assumed fraction of peak FLOPs. Measure it before anyone commits to Phase 3.
#SBATCH --job-name=skilldag-p1-smoke
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=phase1_out/slurm-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p phase1_out

source "$HOME/orcd/pool/venv_skilldag/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

DATA=${DATA:-$HOME/orcd/scratch/skilldag/dolma_domains}
RUNS=${RUNS:-$HOME/orcd/scratch/skilldag/runs}
BUDGET=${BUDGET:-20000000}        # 20M tokens -- enough to measure throughput, not to learn
EVAL_EVERY=${EVAL_EVERY:-5000000}
CKPT_EVERY=${CKPT_EVERY:-10000000}

if [ ! -d "$DATA" ]; then
  echo "ERROR: $DATA not found. Run Phase 0 first (sbatch orcd_phase0_prep.sh)."
  exit 1
fi

common=(--data "$DATA" --weights natural --reweight-mode fixed
        --token-budget "$BUDGET" --eval-tokens "$EVAL_EVERY"
        --ckpt-every-tokens "$CKPT_EVERY" --ckpt-every-seconds 0)

echo
echo "=== A: with gradient checkpointing (current default) ==="
python train_mixture.py --arm smoke --out "$RUNS/smoke_gc" "${common[@]}"

echo
echo "=== B: resume check -- must continue, not restart from 0 ==="
# Same budget: a correct resume finishes immediately at the existing token count.
# If this retrains from 0, checkpoint/resume is broken and every long run is at risk.
python train_mixture.py --arm smoke --out "$RUNS/smoke_gc" "${common[@]}" --resume

echo
echo "=== collecting ==="
mkdir -p phase1_out
cp "$RUNS/smoke_gc/train_log.jsonl" "$RUNS/smoke_gc/val_log.jsonl" phase1_out/ 2>/dev/null || true
cp "$RUNS/smoke_gc/run_config.json" phase1_out/ 2>/dev/null || true

python - "$RUNS/smoke_gc/train_log.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r.get("elapsed_s")]
if len(rows) >= 2:
    a, b = rows[0], rows[-1]
    dt, dtok = b["elapsed_s"] - a["elapsed_s"], b["tokens"] - a["tokens"]
    if dt > 0:
        tps = dtok / dt
        # 8ND with gradient checkpointing on a 1.18B model
        tflops = 8 * 1.18e9 * tps / 1e12
        print(f"\nMEASURED: {tps:,.0f} tokens/s  ~= {tflops:.0f} TFLOP/s sustained")
        print(f"  -> one 2B-token main run: {2e9/tps/3600:.1f} GPU-hours")
        print(f"  -> all 15 main runs:      {15*2e9/tps/3600:.0f} GPU-hours")
EOF

echo
echo "PHASE 1 COMPLETE — send back phase1_out/ plus the MEASURED line above."
echo "That number replaces every estimate in RUNBOOK.md."
echo "Do NOT continue to Phase 2 until the review questions in PLAN.md are settled --"
echo "item 13 especially."
