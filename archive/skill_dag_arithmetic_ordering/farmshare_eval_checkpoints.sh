#!/bin/bash
# Eval-only launcher: submits eval_mastery.py against whatever checkpoints already
# exist in each given run directory. Does NOT resume/continue training -- for runs
# that were cancelled mid-flight, this just measures what was saved before the cancel.
# eval_mastery.py is resume-safe (skips tokens already in eval_log.jsonl), so this is
# also safe to re-run later if more checkpoints show up for the same run names.
#
# Usage (run ON the cluster, from skill_dag/):
#   bash farmshare_eval_checkpoints.sh topo_101 topo_102 topo_103 random_201
set -e
cd "$(dirname "$0")"
[ "$#" -ge 1 ] || { echo "usage: $0 <run_name> [run_name...]"; exit 1; }
GPU_PARTITION=$(sinfo -h -o "%P %G" | awk 'tolower($1) ~ /gpu/ && $2 ~ /gpu/ {print $1; exit}' | tr -d '*')
mkdir -p eval_out
for s in "$@"; do
  if [ ! -d "runs/$s" ]; then
    echo "SKIP $s: runs/$s not found"
    continue
  fi
cat > eval_${s}.sbatch <<JOB
#!/bin/bash
#SBATCH --job-name=ev-${s}
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=eval_out/slurm-${s}-%j.out
source ../.venv_skilldag/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e
python eval_mastery.py --run runs/${s}
echo "EVAL ${s} COMPLETE"
JOB
  sbatch --qos=gpu eval_${s}.sbatch 2>/dev/null || sbatch eval_${s}.sbatch
done
echo "eval jobs submitted for: $*"
echo "watch: squeue -u \$USER"
echo "results land in runs/<name>/eval_log.jsonl as each job completes"
