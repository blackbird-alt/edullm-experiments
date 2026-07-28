#!/bin/bash
# Phase 2 launcher: submits all six main runs in parallel. Usage:
#   bash skill_dag/farmshare_phase2.sh <TOKEN_BUDGET>
# ONLY run after the prereg (with this budget) is committed + posted.
set -e
cd "$(dirname "$0")"
B=${1:?usage: farmshare_phase2.sh TOKEN_BUDGET}
BASE=$HOME/base/olmo-ladder-760m-05xc
GPU_PARTITION=$(sinfo -h -o "%P %G" | awk 'tolower($1) ~ /gpu/ && $2 ~ /gpu/ {print $1; exit}' | tr -d '*')
mkdir -p phase2_out
for s in topo_101 topo_102 topo_103 random_201 random_202 random_203; do
cat > phase2_${s}.sbatch <<JOB
#!/bin/bash
#SBATCH --job-name=sd-${s}
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=phase2_out/slurm-${s}-%j.out
source ../.venv_skilldag/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e
python train_cpt.py --model $BASE --schedule schedules/${s}.idx \
  --token-budget $B --ckpt-tokens $((B/12)) --out runs/${s}
python eval_mastery.py --run runs/${s}
echo "RUN ${s} COMPLETE"
JOB
sbatch --qos=gpu phase2_${s}.sbatch 2>/dev/null || sbatch phase2_${s}.sbatch
done
echo "six runs submitted at budget $B; watch: squeue -u \$USER"
