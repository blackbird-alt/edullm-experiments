#!/bin/bash
# Phase 1 bootstrap for FarmShare (or any Slurm cluster with a GPU).
# Run this ON THE CLUSTER after logging in:
#   bash skill_dag/farmshare_phase1.sh
# It checks for GPUs, sets up a venv, snapshots the base model, and submits
# the pilot + smoke run as one GPU job. Outputs land in skill_dag/phase1_out/.
set -e
cd "$(dirname "$0")"

echo "=== GPU availability on this cluster ==="
sinfo -o "%P %G %D"
# prefer a partition literally named *gpu*; else any partition whose GRES has gpus
GPU_PARTITION=$(sinfo -h -o "%P %G" | awk 'tolower($1) ~ /gpu/ && $2 ~ /gpu/ {print $1; exit}' | tr -d '*')
[ -z "$GPU_PARTITION" ] && GPU_PARTITION=$(sinfo -h -o "%P %G" | awk '$2 ~ /gpu/ {print $1; exit}' | tr -d '*')
if [ -z "$GPU_PARTITION" ]; then
  echo "NO GPU PARTITION VISIBLE. Phase 1 needs one GPU."
  echo "If this cluster has no GPUs, this same script works on the team cluster."
  exit 1
fi
echo "using partition: $GPU_PARTITION"

echo "=== environment (one-time, ~5 min) ==="
module load python 2>/dev/null || true
if [ ! -d ../.venv_skilldag ]; then
  python3 -m venv ../.venv_skilldag
fi
source ../.venv_skilldag/bin/activate
pip install --quiet torch transformers "huggingface_hub[cli]" wandb

echo "=== snapshot base model (pinned revision, ~3.6 GB) ==="
BASE=$HOME/base/olmo-ladder-760m-05xc
if [ ! -f "$BASE/model.safetensors" ]; then
  hf download allenai/OLMo-Ladder-760M-0.5xC \
    --revision c8f6875e640a985de0709e8be1f38eef0cb4bc48 \
    --local-dir "$BASE"
fi
echo "base at $BASE"

echo "=== submitting Phase 1 GPU job ==="
mkdir -p phase1_out
cat > phase1_job.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=skilldag-p1
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=phase1_out/slurm-%j.out
nvidia-smi -L || true
source ../.venv_skilldag/bin/activate
set -e
echo "--- pilot ---"
python pilot_calibrate.py --model $BASE
echo "--- smoke run (200M tokens) ---"
python train_cpt.py --model $BASE \\
  --schedule schedules/random_201.idx --token-budget 200000000 \\
  --ckpt-tokens 25000000 --out runs/smoke_random201
echo "--- eval smoke checkpoints ---"
python eval_mastery.py --run runs/smoke_random201
echo "--- collecting results ---"
cp pilot_results.json phase1_out/
cp runs/smoke_random201/eval_log.jsonl phase1_out/
cp runs/smoke_random201/train_log.jsonl phase1_out/
echo "PHASE 1 COMPLETE - send back the 3 files in skill_dag/phase1_out/"
EOF
# FarmShare's gpu partition historically requires --qos=gpu; try with it, fall back without
sbatch --qos=gpu phase1_job.sbatch 2>/dev/null || sbatch phase1_job.sbatch
echo
echo "Job submitted. Check with: squeue -u \$USER"
echo "When done, the 3 result files are in skill_dag/phase1_out/ - copy them back with:"
echo "  scp farmshare:$(pwd)/phase1_out/* ."
