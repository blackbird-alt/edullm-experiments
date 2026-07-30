#!/bin/bash
# Phase 2: the fitting runs that produce the weights for arms 2-5. ~35 GPU-h total.
#   NSHARDS=8 I_HAVE_SETTLED_ITEM_13=yes bash farmshare_phase2_fit.sh
#
# This is a SUBMITTER, not a job -- run it on the login node. It submits six jobs
# with the right dependencies:
#
#   fleet array (96 runs, GPU) --> fleet merge (CPU) --> fitters (CPU) --> clusters
#                                                                            |
#   arm4 array (45 runs, GPU) --> arm4 merge (CPU)                           v
#     (independent; feeds nothing here)          arm5 array (10 runs, GPU) --> arm5 merge
#
# The GPU work is split NSHARDS ways: each array task runs every NSHARDS'th probe and
# appends to its own log, then a short CPU job merges the logs and writes the artifact.
# Taken sequentially the fleet is ~20 h and arm 4 ~13 h of wall clock; at NSHARDS=8 both
# drop to a couple of hours, subject to how many GPUs you can actually hold at once.
# NSHARDS=1 reproduces the old sequential behaviour exactly.
#
# Everything is resumable at run granularity, so a task killed by the time limit loses
# at most the probe it was in the middle of. Resubmitting the same command picks up.
#
# DO NOT RUN until the review questions in PLAN.md are settled. Item 13 in
# particular: the A_ij probe holds total tokens fixed rather than j's, which biases
# every entry negative and can freeze the adaptive arms at their starting weights.
# That produces a clean-looking null that means nothing, and it costs 55 probe runs
# plus six main runs to discover afterwards.
set -e
cd "$(dirname "$0")"
mkdir -p phase2_out

if [ "${I_HAVE_SETTLED_ITEM_13}" != "yes" ]; then
  echo "Refusing to run: item 13 (A_ij token matching) is unresolved."
  echo "See 'Questions for review' in PLAN.md. If it has been decided, rerun with:"
  echo "  I_HAVE_SETTLED_ITEM_13=yes bash farmshare_phase2_fit.sh"
  exit 1
fi

NSHARDS="${NSHARDS:-8}"
# Generous enough that NSHARDS=1 still fits; lower it when NSHARDS is large, since a
# shorter request gets backfilled onto a GPU sooner.
GPU_TIME="${GPU_TIME:-48:00:00}"
# Append e.g. %4 to cap how many array tasks run at once if the partition is tight.
THROTTLE="${THROTTLE:-}"

GPU_ARGS="--gres=gpu:1 --cpus-per-task=8 --mem=48G"
CPU_ARGS="--cpus-per-task=4 --mem=16G"
PRE='cd $SLURM_SUBMIT_DIR; source ../.venv_skilldag/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e'
LAST=$((NSHARDS - 1))
# arm 5 has only 10 probes, so more shards than that would idle whole GPUs
A5=$((NSHARDS < 10 ? NSHARDS : 10))
A5_LAST=$((A5 - 1))

echo "sharding GPU work ${NSHARDS} ways (arm 5: ${A5}); time limit ${GPU_TIME}"
echo

# --- 2a: shared proxy fleet for arms 2 and 3 (96 runs) ---
FLEET=$(sbatch --parsable $GPU_ARGS --time=$GPU_TIME --array=0-${LAST}${THROTTLE} \
  --job-name=skilldag-p2-fleet --output=phase2_out/fleet-%A_%a.out \
  --wrap "$PRE; python fit_proxy_fleet.py --out fleet --mixtures 32 --sizes 50 75 100 \
          --shard \$SLURM_ARRAY_TASK_ID --num-shards $NSHARDS")
echo "fleet          -> job $FLEET  (96 runs over $NSHARDS tasks)"

# Merging is separate rather than left to whichever task finishes last: on a shared
# filesystem that task may not see its siblings' final writes yet.
FLEETM=$(sbatch --parsable $CPU_ARGS --time=00:30:00 --dependency=afterok:$FLEET \
  --job-name=skilldag-p2-fleet-merge --output=phase2_out/fleet-merge-%j.out \
  --wrap "$PRE; python fit_proxy_fleet.py --out fleet --mixtures 32 --sizes 50 75 100 \
          --assemble-only")
echo "fleet merge    -> job $FLEETM  (writes fleet/fleet.jsonl)"

# --- 2b: arm 4 full pairwise A_ij (45 runs). Independent of the fleet. ---
AIJ4=$(sbatch --parsable $GPU_ARGS --time=$GPU_TIME --array=0-${LAST}${THROTTLE} \
  --job-name=skilldag-p2-aij4 --output=phase2_out/aij4-%A_%a.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm4 \
          --shard \$SLURM_ARRAY_TASK_ID --num-shards $NSHARDS")
echo "arm4 A_ij      -> job $AIJ4  (45 runs over $NSHARDS tasks, parallel with fleet)"

AIJ4M=$(sbatch --parsable $CPU_ARGS --time=00:30:00 --dependency=afterok:$AIJ4 \
  --job-name=skilldag-p2-aij4-merge --output=phase2_out/aij4-merge-%j.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm4 --assemble-only")
echo "arm4 merge     -> job $AIJ4M  (writes aij_arm4/aij.json)"

# --- 3+4: CPU fitters and clustering, after the fleet is merged ---
FIT=$(sbatch --parsable $CPU_ARGS --time=02:00:00 --dependency=afterok:$FLEETM \
  --job-name=skilldag-p2-fitters --output=phase2_out/fitters-%j.out \
  --wrap "$PRE; \
    python fit_regmix.py --fleet fleet --out weights_arm2.json; \
    python fit_mixing_law.py --fleet fleet --out weights_arm3.json --t-out mixlaw_t.json; \
    python cluster_tlite.py --mixing-law mixlaw_t.json --k 4 --out clusters.json")
echo "fitters+cluster-> job $FIT  (after fleet merge)"

# --- 5: arm 5 cluster-level A_ij (10 runs), after clusters exist ---
AIJ5=$(sbatch --parsable $GPU_ARGS --time=12:00:00 --array=0-${A5_LAST}${THROTTLE} \
  --dependency=afterok:$FIT \
  --job-name=skilldag-p2-aij5 --output=phase2_out/aij5-%A_%a.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm5 --cluster-map clusters.json \
          --shard \$SLURM_ARRAY_TASK_ID --num-shards $A5")
echo "arm5 A_ij      -> job $AIJ5  (after clusters, 10 runs over $A5 tasks)"

AIJ5M=$(sbatch --parsable $CPU_ARGS --time=00:30:00 --dependency=afterok:$AIJ5 \
  --job-name=skilldag-p2-aij5-merge --output=phase2_out/aij5-merge-%j.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm5 --cluster-map clusters.json --assemble-only")
echo "arm5 merge     -> job $AIJ5M  (writes aij_arm5/aij.json)"

cat <<'EOF'

Submitted. Watch with: squeue -u $USER

If a merge job fails because tasks are still outstanding, it prints how many are left.
Rerun the array with the same command to fill the gaps -- completed runs are skipped --
then rerun the merge with --assemble-only.

WHEN THE PROBES FINISH, CHECK THIS BEFORE TRUSTING ANYTHING:

  python -c "import json; a=json.load(open('aij_arm4/aij.json'))['A']; \
  v=[x for r in a.values() for x in r.values()]; \
  print('positive:', sum(1 for x in v if x>0), '/', len(v))"

0 positive entries means A is degenerate and arms 4 and 5 are meaningless. Two
different causes with two different fixes: probes too short (review item 5), or the
token-matching confound (item 13). Do not proceed to Phase 3 on a degenerate matrix.
EOF
