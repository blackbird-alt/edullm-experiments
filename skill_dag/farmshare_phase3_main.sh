#!/bin/bash
# Phase 3: the 15 main runs. ~490 GPU-h. Slurm array, one task per run.
#   sbatch farmshare_phase3_main.sh
#
# REQUIRES THE PREREG TO BE FILED. Program rule: arms and thresholds are registered
# before the real spend. PREREG.md still has 9 [SET BEFORE LAUNCH] values.
#
# All 15 runs are independent and resume-safe. If the array is cut off, resubmit --
# --resume reloads model, optimizer, scheduler, step, token count, per-domain read
# cursors, wrap counts, current weights, and the next-eval/next-round boundaries.
#SBATCH --job-name=skilldag-p3-main
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=phase3_out/run-%A_%a.out
set -e
cd "$(dirname "$0")"
mkdir -p phase3_out

source ../.venv_skilldag/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BUDGET=${BUDGET:-2000000000}          # tokens per run -- [SET BEFORE LAUNCH] in PREREG.md
N_SEEDS=3

# The source doc says arms 4/5 "select optimal domain weight to start, then adjust 5
# times" without saying whether "optimal to start" is the natural mix or arm 2's fitted
# optimum. Default is natural; export ADAPTIVE_INIT=weights_arm2.json for the other
# reading. Whichever you pick, record it in the prereg and do not change it after
# seeing results.
ADAPTIVE_INIT=${ADAPTIVE_INIT:-natural}

for f in weights_arm2.json weights_arm3.json aij_arm4/aij.json aij_arm5/aij.json clusters.json; do
  [ -e "$f" ] || { echo "ERROR: missing $f -- run Phase 2 first."; exit 1; }
done

# 5 arms x 3 seeds, indexed 0..14
ARMS=(arm1_natural arm2_regmix arm3_mixlaw arm4_skillit arm5_tlite)
ARM_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
ARM=${ARMS[$ARM_IDX]}
OUT="runs/${ARM}_seed${SEED_IDX}"

case "$ARM" in
  arm1_natural) EXTRA="--weights natural                --reweight-mode fixed" ;;
  arm2_regmix)  EXTRA="--weights weights_arm2.json      --reweight-mode fixed" ;;
  arm3_mixlaw)  EXTRA="--weights weights_arm3.json      --reweight-mode fixed" ;;
  arm4_skillit) EXTRA="--weights ${ADAPTIVE_INIT} --reweight-mode adaptive --aij aij_arm4/aij.json" ;;
  arm5_tlite)   EXTRA="--weights ${ADAPTIVE_INIT} --reweight-mode adaptive --aij aij_arm5/aij.json --cluster-map clusters.json" ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${ARM} seed ${SEED_IDX} -> ${OUT} ==="
nvidia-smi -L || true

# --seed-index / --n-seeds give each replicate a disjoint region of every domain pool,
# so the three seeds are independent replicates rather than the same tokens reshuffled.
python train_mixture.py \
  --arm "$ARM" --out "$OUT" \
  $EXTRA \
  --token-budget "$BUDGET" \
  --seed "$SEED_IDX" --seed-index "$SEED_IDX" --n-seeds "$N_SEEDS" \
  --rounds 5 \
  --resume

echo "done: $OUT"
echo
echo "When all 15 finish, run Phase 4 on a CPU node:"
echo "  python analyze.py --runs \"runs/arm*\" --margin <PREREG margin> \\"
echo "    --fitting-costs aij_arm4/aij.json aij_arm5/aij.json fleet/fleet_cost.json \\"
echo "    --out analysis.json"
