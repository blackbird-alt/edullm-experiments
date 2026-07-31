#!/bin/bash
# Phase 2: the fitting runs that produce the weights for arms 2-5.
#   NSHARDS=8 I_HAVE_SETTLED_ITEM_13=yes bash orcd_phase2_fit.sh
#
# Run this with `bash`, not `sbatch` -- it is a submitter. It submits six jobs with the
# right dependencies:
#
#   fleet array (96 runs, GPU) --> fleet merge (CPU) --> fitters (CPU) --> clusters
#                                                                            |
#   arm4 array (45 runs, GPU) --> arm4 merge (CPU)                           v
#     (independent; feeds nothing here)          arm5 array (10 runs, GPU) --> arm5 merge
#
# Each probe is a 50-100M proxy on 200M tokens: roughly 30 min on an L40S, 10 on an
# H200. So even the 6 h cap on mit_normal_gpu fits ~12 probes per job, and the scripts
# resume at probe granularity -- a task killed by the time limit loses at most the one
# probe it was in the middle of. Resubmit the same command to fill any gaps.
#
# NSHARDS is how many GPUs to spread each array over. mit_normal_gpu allows 2 at once
# and mit_preemptable 4, so setting it higher just queues; it does not break anything.
#
# DO NOT RUN until the review questions in PLAN.md are settled. Item 13 in particular:
# the A_ij probe holds total tokens fixed rather than j's, which biases every entry
# negative and can freeze the adaptive arms at their starting weights. That produces a
# clean-looking null that means nothing, and it costs 55 probe runs plus six main runs
# to discover afterwards.
set -e
cd "$(dirname "$0")"
mkdir -p phase2_out

if [ "${I_HAVE_SETTLED_ITEM_13}" != "yes" ]; then
  echo "Refusing to run: item 13 (A_ij token matching) is unresolved."
  echo "See 'Questions for review' in PLAN.md. If it has been decided, rerun with:"
  echo "  I_HAVE_SETTLED_ITEM_13=yes bash orcd_phase2_fit.sh"
  exit 1
fi

PARTITION=${PARTITION:-mit_preemptable}
GPU=${GPU:-l40s}
NSHARDS=${NSHARDS:-4}
DATA=${DATA:-$HOME/orcd/scratch/skilldag/dolma_domains}

case "$PARTITION" in
  mit_normal_gpu)  WALL=${WALL:-6:00:00} ;;
  mit_preemptable) WALL=${WALL:-48:00:00} ;;
  # PI/group partition: PARTITION=pi_yourgroup WALL=7-00:00:00 GPU=h100
  *)               WALL=${WALL:-12:00:00} ;;
esac
# Append e.g. %2 to cap concurrent array tasks; mit_normal_gpu allows 2 GPUs.
THROTTLE="${THROTTLE:-}"

# Public partitions take -G type:count; group partitions generally expect --gres, and
# asking them for a GPU type they do not advertise leaves the job pending forever.
if [ -z "${GPU_REQ:-}" ]; then
  case "$PARTITION" in
    mit_normal_gpu|mit_preemptable) GPU_REQ="-G ${GPU}:1" ;;
    *)                              GPU_REQ="--gres=gpu:1" ;;
  esac
fi
read -ra GPU_REQ_ARR <<< "$GPU_REQ"

GPU_ARGS=(--partition="$PARTITION" "${GPU_REQ_ARR[@]}" --cpus-per-task=16 --mem=64G
          --time="$WALL" --requeue)
CPU_ARGS=(--partition=mit_normal --cpus-per-task=8 --mem=32G)
PRE="cd \$SLURM_SUBMIT_DIR; source \$HOME/orcd/pool/venv_skilldag/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e"
DFLAG="--data $DATA"
LAST=$((NSHARDS - 1))
# arm 5 has only 10 probes, so more shards than that would idle whole GPUs
A5=$((NSHARDS < 10 ? NSHARDS : 10))
A5_LAST=$((A5 - 1))

echo "partition=$PARTITION  request='${GPU_REQ}'  wall=$WALL  shards=$NSHARDS (arm5: $A5)"
echo

# --- 2a: shared proxy fleet for arms 2 and 3 (96 runs) ---
FLEET=$(sbatch --parsable "${GPU_ARGS[@]}" --array=0-${LAST}${THROTTLE} \
  --job-name=skilldag-p2-fleet --output=phase2_out/fleet-%A_%a.out \
  --wrap "$PRE; python fit_proxy_fleet.py --out fleet $DFLAG --mixtures 32 --sizes 50 75 100 \
          --shard \$SLURM_ARRAY_TASK_ID --num-shards $NSHARDS")
echo "fleet          -> job $FLEET  (96 runs over $NSHARDS tasks)"

# Merging is separate rather than left to whichever task finishes last: on a shared
# filesystem that task may not see its siblings' final writes yet.
FLEETM=$(sbatch --parsable "${CPU_ARGS[@]}" --time=00:30:00 --dependency=afterok:$FLEET \
  --job-name=skilldag-p2-fleet-merge --output=phase2_out/fleet-merge-%j.out \
  --wrap "$PRE; python fit_proxy_fleet.py --out fleet $DFLAG --mixtures 32 --sizes 50 75 100 \
          --assemble-only")
echo "fleet merge    -> job $FLEETM  (writes fleet/fleet.jsonl)"

# --- 2b: arm 4 full pairwise A_ij (45 runs). Independent of the fleet. ---
AIJ4=$(sbatch --parsable "${GPU_ARGS[@]}" --array=0-${LAST}${THROTTLE} \
  --job-name=skilldag-p2-aij4 --output=phase2_out/aij4-%A_%a.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm4 $DFLAG \
          --shard \$SLURM_ARRAY_TASK_ID --num-shards $NSHARDS")
echo "arm4 A_ij      -> job $AIJ4  (45 runs over $NSHARDS tasks, parallel with fleet)"

AIJ4M=$(sbatch --parsable "${CPU_ARGS[@]}" --time=00:30:00 --dependency=afterok:$AIJ4 \
  --job-name=skilldag-p2-aij4-merge --output=phase2_out/aij4-merge-%j.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm4 $DFLAG --assemble-only")
echo "arm4 merge     -> job $AIJ4M  (writes aij_arm4/aij.json)"

# --- 3+4: CPU fitters and clustering, after the fleet is merged ---
FIT=$(sbatch --parsable "${CPU_ARGS[@]}" --time=02:00:00 --dependency=afterok:$FLEETM \
  --job-name=skilldag-p2-fitters --output=phase2_out/fitters-%j.out \
  --wrap "$PRE; \
    python fit_regmix.py --fleet fleet --out weights_arm2.json; \
    python fit_mixing_law.py --fleet fleet --out weights_arm3.json --t-out mixlaw_t.json; \
    python cluster_tlite.py --mixing-law mixlaw_t.json --k 4 --out clusters.json")
echo "fitters+cluster-> job $FIT  (after fleet merge)"

# --- 5: arm 5 cluster-level A_ij (10 runs), after clusters exist ---
AIJ5=$(sbatch --parsable "${GPU_ARGS[@]}" --array=0-${A5_LAST}${THROTTLE} \
  --dependency=afterok:$FIT \
  --job-name=skilldag-p2-aij5 --output=phase2_out/aij5-%A_%a.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm5 $DFLAG --cluster-map clusters.json \
          --shard \$SLURM_ARRAY_TASK_ID --num-shards $A5")
echo "arm5 A_ij      -> job $AIJ5  (after clusters, 10 runs over $A5 tasks)"

AIJ5M=$(sbatch --parsable "${CPU_ARGS[@]}" --time=00:30:00 --dependency=afterok:$AIJ5 \
  --job-name=skilldag-p2-aij5-merge --output=phase2_out/aij5-merge-%j.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm5 $DFLAG --cluster-map clusters.json --assemble-only")
echo "arm5 merge     -> job $AIJ5M  (writes aij_arm5/aij.json)"

cat <<'EOF'

Submitted. Watch with: squeue -u $USER

On mit_preemptable a killed task is requeued and resumes at probe granularity. If a
merge job reports outstanding runs, resubmit the array with the same command to fill
the gaps -- completed probes are skipped -- then rerun the merge with --assemble-only.

WHEN THE PROBES FINISH, CHECK THIS BEFORE TRUSTING ANYTHING:

  python -c "import json; a=json.load(open('aij_arm4/aij.json'))['A']; \
  v=[x for r in a.values() for x in r.values()]; \
  print('positive:', sum(1 for x in v if x>0), '/', len(v))"

0 positive entries means A is degenerate and arms 4 and 5 are meaningless. Two
different causes with two different fixes: probes too short (review item 5), or the
token-matching confound (item 13). Do not proceed to Phase 3 on a degenerate matrix.
EOF
