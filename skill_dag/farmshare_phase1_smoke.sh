#!/bin/bash
# Phase 1: FIRST CONTACT WITH A GPU. ~10 minutes, one GPU.
#   sbatch farmshare_phase1_smoke.sh
#
# Nothing in this repo has ever run on a GPU or read a real token -- every check so
# far was synthetic data on CPU. This run answers the questions that cannot be
# answered any other way:
#   - does OLMo-1B load at revision step1000-tokens4B?
#   - does it fit in memory at batch 8 x accum 8?
#   - what is the ACTUAL throughput vs the assumed 1.2e14 FLOP/s the 525 GPU-h
#     estimate is built on?
#   - does checkpoint/resume round-trip real model weights?
#
# Run this before committing to anything else. It is cheap and it is the only
# thing standing between you and discovering an OOM 300 GPU-hours in.
#SBATCH --job-name=skilldag-p1-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=phase1_out/slurm-%j.out
set -e
cd "$(dirname "$0")"
mkdir -p phase1_out

source ../.venv_skilldag/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi -L || true
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

DATA=${DATA:-dolma_domains}
BUDGET=${BUDGET:-20000000}        # 20M tokens -- enough to measure throughput, not to learn
EVAL_EVERY=${EVAL_EVERY:-5000000}
CKPT_EVERY=${CKPT_EVERY:-10000000}

if [ ! -d "$DATA" ]; then
  echo "ERROR: $DATA/ not found. Run Phase 0 first (farmshare_phase0_prep.sh)."
  exit 1
fi

echo "=== smoke run: ${BUDGET} tokens, fixed natural weights ==="
python train_mixture.py \
  --arm smoke --out runs/smoke \
  --data "$DATA" \
  --weights natural --reweight-mode fixed \
  --token-budget "$BUDGET" \
  --eval-tokens "$EVAL_EVERY" \
  --ckpt-every-tokens "$CKPT_EVERY"

echo
echo "=== resume check: rerun with --resume, must continue rather than restart ==="
# Same budget: a correct resume finishes immediately at the existing token count.
# If this retrains from 0, checkpoint/resume is broken and every long run is at risk.
python train_mixture.py \
  --arm smoke --out runs/smoke \
  --data "$DATA" \
  --weights natural --reweight-mode fixed \
  --token-budget "$BUDGET" \
  --eval-tokens "$EVAL_EVERY" \
  --ckpt-every-tokens "$CKPT_EVERY" \
  --resume

echo
echo "=== collecting ==="
cp runs/smoke/train_log.jsonl runs/smoke/val_log.jsonl phase1_out/ 2>/dev/null || true
cp runs/smoke/run_config.json phase1_out/ 2>/dev/null || true

echo
echo "PHASE 1 COMPLETE — send back the files in phase1_out/ plus the tokens/sec figure."
echo "We convert throughput into a real wall-clock number before anyone commits to"
echo "the 490 GPU-h of main runs. Do NOT continue to Phase 2 until the review"
echo "questions in PLAN.md are settled -- item 13 especially."
