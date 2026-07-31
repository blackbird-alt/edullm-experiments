#!/bin/bash
# Phase 3: the 15 main runs.
#   PARTITION=pi_yourgroup WALL=7-00:00:00 CONC=4 bash orcd_phase3_main.sh
#   bash orcd_phase3_main.sh                    # public fallback: mit_preemptable
#
# Run this with `bash`, not `sbatch`. It submits itself as an array and each task
# resubmits itself until its run finishes.
#
# WHY THE CHAIN: one 2B-token run on a 1.18B model is ~27 GPU-h on an H100, which fits in
# a single job on a group partition but in neither ORCD public window (6 h on
# mit_normal_gpu, 48 h on mit_preemptable). So each task trains up to --max-seconds,
# checkpoints, exits, and resubmits itself. On owned nodes with a multi-day limit it
# should never need to. train_mixture.py writes run_config.json only on completion, which
# is how a task knows whether it is done.
#
# Preemption is handled separately: --requeue lets Slurm restart a preempted task, and
# --signal=USR1@180 gives the trainer three minutes to checkpoint before it is killed.
# Between the two, the worst case loss is one checkpoint interval (default 30 min).
#
# REQUIRES THE PREREG TO BE FILED. PREREG.md still has 9 [SET BEFORE LAUNCH] values.
set -e
cd "$(dirname "$0")"

PARTITION=${PARTITION:-mit_preemptable}
GPU=${GPU:-h100}                      # only used on the public partitions, which take -G
DATA=${DATA:-$HOME/orcd/scratch/skilldag/dolma_domains}
RUNS=${RUNS:-$HOME/orcd/scratch/skilldag/runs}
BUDGET=${BUDGET:-2000000000}          # tokens per run -- [SET BEFORE LAUNCH] in PREREG.md
N_SEEDS=${N_SEEDS:-3}
# Eight evaluations per run, scaled off the budget rather than fixed. analyze.py needs a
# target to be held for two consecutive evaluations before it counts as crossed, so the
# number of eval points sets the resolution of the whole comparison. A fixed cadence would
# quietly halve that resolution the moment someone halves BUDGET.
EVAL_EVERY=${EVAL_EVERY:-$((BUDGET / 8))}
MAX_CHUNKS=${MAX_CHUNKS:-40}          # stop a runaway resubmission loop

case "$PARTITION" in
  mit_normal_gpu)  WALL=${WALL:-6:00:00};  CONC=${CONC:-2} ;;
  mit_preemptable) WALL=${WALL:-48:00:00}; CONC=${CONC:-4} ;;
  # A PI/group partition: set WALL to its limit and CONC to how many GPUs you own.
  # e.g. PARTITION=pi_yourgroup WALL=7-00:00:00 CONC=4 GPU=h100
  *)               WALL=${WALL:-12:00:00}; CONC=${CONC:-4} ;;
esac

# Train for the job's limit minus a margin for model load, the final checkpoint write and
# Slurm overhead, then stop cleanly. Derived rather than hardcoded so that raising WALL on
# a group partition actually lengthens the chunks instead of silently keeping short ones.
slurm_time_s() {
  local t=$1 d=0 rest=$1 p
  case "$t" in *-*) d=${t%%-*}; rest=${t#*-} ;; esac
  IFS=: read -ra p <<< "$rest"
  case "${#p[@]}:$t" in
    1:*-*) echo $(( d * 86400 + 10#${p[0]} * 3600 )) ;;
    1:*)   echo $(( 10#${p[0]} * 60 )) ;;
    2:*-*) echo $(( d * 86400 + 10#${p[0]} * 3600 + 10#${p[1]} * 60 )) ;;
    2:*)   echo $(( 10#${p[0]} * 60 + 10#${p[1]} )) ;;
    *)     echo $(( d * 86400 + 10#${p[0]} * 3600 + 10#${p[1]} * 60 + 10#${p[2]} )) ;;
  esac
}
MARGIN=${MARGIN:-1200}
SOFT=${SOFT:-$(( $(slurm_time_s "$WALL") - MARGIN ))}
[ "$SOFT" -gt 0 ] || { echo "ERROR: WALL=$WALL is shorter than MARGIN=${MARGIN}s."; exit 1; }

# The public partitions document -G type:count. PI/group partitions generally predate that
# and expect --gres; asking for a type they do not advertise leaves the job pending forever.
if [ -z "${GPU_REQ:-}" ]; then
  case "$PARTITION" in
    mit_normal_gpu|mit_preemptable) GPU_REQ="-G ${GPU}:1" ;;
    *)                              GPU_REQ="--gres=gpu:1" ;;
  esac
fi
read -ra GPU_REQ_ARR <<< "$GPU_REQ"

SB_FLAGS=(--partition="$PARTITION" "${GPU_REQ_ARR[@]}" --cpus-per-task=16 --mem=64G
          --time="$WALL" --requeue --signal=USR1@180
          --job-name=skilldag-p3 --output=phase3_out/run-%A_%a.out)

# ---------------------------------------------------------------- status
if [ "$1" = "--status" ]; then
  printf '%-24s %14s %8s %7s  %s\n' ARM TOKENS PCT CHUNKS STATE
  done_n=0; total=0
  for i in $(seq 0 $((5 * N_SEEDS - 1))); do
    ARMS=(arm1_natural arm2_regmix arm3_mixlaw arm4_skillit arm5_tlite)
    a=${ARMS[$((i / N_SEEDS))]}; s=$((i % N_SEEDS)); d="$RUNS/${a}_seed${s}"
    total=$((total + 1))
    c=$(cat "$d/.chunks" 2>/dev/null || echo 0)
    if [ -e "$d/run_config.json" ]; then
      t=$(python -c "import json;print(json.load(open('$d/run_config.json'))['final_tokens'])" 2>/dev/null || echo '?')
      printf '%-24s %14s %7s%% %7s  %s\n' "${a}_seed${s}" "$t" 100 "$c" COMPLETE
      done_n=$((done_n + 1))
    elif [ -e "$d/ckpt_resume/progress.json" ]; then
      t=$(python -c "import json;print(json.load(open('$d/ckpt_resume/progress.json'))['trained'])" 2>/dev/null || echo 0)
      printf '%-24s %14s %7.1f%% %7s  %s\n' "${a}_seed${s}" "$t" \
        "$(python -c "print(100*$t/$BUDGET)" 2>/dev/null || echo 0)" "$c" running
    else
      printf '%-24s %14s %8s %7s  %s\n' "${a}_seed${s}" 0 '-' "$c" 'not started'
    fi
  done
  echo
  echo "$done_n/$total complete"
  exit 0
fi

# ---------------------------------------------------------------- submit mode
if [ -z "$SLURM_JOB_ID" ]; then
  mkdir -p phase3_out
  for f in weights_arm2.json weights_arm3.json aij_arm4/aij.json aij_arm5/aij.json clusters.json; do
    [ -e "$f" ] || { echo "ERROR: missing $f -- run Phase 2 first."; exit 1; }
  done
  [ -d "$DATA" ] || { echo "ERROR: $DATA not found -- run Phase 0 first."; exit 1; }
  echo "partition=$PARTITION  request='${GPU_REQ}'  wall=$WALL  train-for=${SOFT}s"
  echo "budget=$BUDGET tokens, eval every $EVAL_EVERY, 5 arms x $N_SEEDS seeds, $CONC at a time"
  sbatch "${SB_FLAGS[@]}" --array=0-$((5 * N_SEEDS - 1))%${CONC} "$0"
  echo
  echo "Each task resubmits itself until its run completes. Watch: squeue -u \$USER"
  echo "Progress across all runs:  bash $0 --status"
  exit 0
fi

# ---------------------------------------------------------------- worker mode
mkdir -p phase3_out
source "$HOME/orcd/pool/venv_skilldag/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The source doc says arms 4/5 "select optimal domain weight to start, then adjust 5
# times" without saying whether "optimal to start" is the natural mix or arm 2's fitted
# optimum. Default is natural; export ADAPTIVE_INIT=weights_arm2.json for the other
# reading. Whichever you pick, record it in the prereg and do not change it after
# seeing results.
ADAPTIVE_INIT=${ADAPTIVE_INIT:-natural}

ARMS=(arm1_natural arm2_regmix arm3_mixlaw arm4_skillit arm5_tlite)
ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
OUT="$RUNS/${ARM}_seed${SEED_IDX}"

case "$ARM" in
  arm1_natural) EXTRA="--weights natural           --reweight-mode fixed" ;;
  arm2_regmix)  EXTRA="--weights weights_arm2.json --reweight-mode fixed" ;;
  arm3_mixlaw)  EXTRA="--weights weights_arm3.json --reweight-mode fixed" ;;
  arm4_skillit) EXTRA="--weights ${ADAPTIVE_INIT} --reweight-mode adaptive --aij aij_arm4/aij.json" ;;
  arm5_tlite)   EXTRA="--weights ${ADAPTIVE_INIT} --reweight-mode adaptive --aij aij_arm5/aij.json --cluster-map clusters.json" ;;
esac

if [ -e "$OUT/run_config.json" ]; then
  echo "$OUT already complete; nothing to do."
  exit 0
fi

CHUNK_F="$OUT/.chunks"
mkdir -p "$OUT"
CHUNK=$(( $(cat "$CHUNK_F" 2>/dev/null || echo 0) + 1 ))
echo "$CHUNK" > "$CHUNK_F"

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${ARM} seed ${SEED_IDX}, chunk ${CHUNK} -> ${OUT} ==="
nvidia-smi -L || true

# --seed-index / --n-seeds give each replicate a disjoint region of every domain pool,
# so the three seeds are independent replicates rather than the same tokens reshuffled.
python train_mixture.py \
  --arm "$ARM" --out "$OUT" --data "$DATA" \
  $EXTRA \
  --token-budget "$BUDGET" --eval-tokens "$EVAL_EVERY" \
  --seed "$SEED_IDX" --seed-index "$SEED_IDX" --n-seeds "$N_SEEDS" \
  --rounds 5 \
  --max-seconds "$SOFT" \
  --resume

if [ -e "$OUT/run_config.json" ]; then
  echo "COMPLETE after ${CHUNK} chunk(s): $OUT"
  exit 0
fi

if [ "$CHUNK" -ge "$MAX_CHUNKS" ]; then
  echo "ERROR: ${ARM} seed ${SEED_IDX} still unfinished after ${MAX_CHUNKS} chunks."
  echo "Something is wrong -- check throughput against the Phase 1 measurement."
  exit 1
fi

echo "Not finished; resubmitting task ${SLURM_ARRAY_TASK_ID} (chunk $((CHUNK + 1)))."
sbatch "${SB_FLAGS[@]}" --array="${SLURM_ARRAY_TASK_ID}" "$0"
