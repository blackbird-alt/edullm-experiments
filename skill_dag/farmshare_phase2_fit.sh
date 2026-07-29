#!/bin/bash
# Phase 2: the fitting runs that produce the weights for arms 2-5. ~35 GPU-h.
#   bash farmshare_phase2_fit.sh
#
# This is a SUBMITTER, not a job -- run it on the login node. It submits four jobs
# with the right dependencies:
#
#   fleet (96 runs, GPU) ---+--> fitters (CPU) --> clusters --> arm5 probe (10 runs, GPU)
#   arm4 probe (45, GPU) ---'  (arm4 runs in parallel; it feeds nothing here)
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

GPU_ARGS="--gres=gpu:1 --cpus-per-task=8 --mem=48G"
PRE='cd $SLURM_SUBMIT_DIR; source ../.venv_skilldag/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e'

# --- 2a: shared proxy fleet for arms 2 and 3 (96 runs, sequential, resumable) ---
# fit_proxy_fleet.py loops internally and skips completed runs on restart, so if this
# hits the time limit just resubmit. It does not shard across nodes.
FLEET=$(sbatch --parsable $GPU_ARGS --time=48:00:00 \
  --job-name=skilldag-p2-fleet --output=phase2_out/fleet-%j.out \
  --wrap "$PRE; python fit_proxy_fleet.py --out fleet --mixtures 32 --sizes 50 75 100")
echo "fleet          -> job $FLEET  (96 runs)"

# --- 2b: arm 4 full pairwise A_ij (45 runs). Independent of the fleet. ---
AIJ4=$(sbatch --parsable $GPU_ARGS --time=48:00:00 \
  --job-name=skilldag-p2-aij4 --output=phase2_out/aij4-%j.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm4")
echo "arm4 A_ij      -> job $AIJ4  (45 runs, parallel with fleet)"

# --- 3+4: CPU fitters and clustering, after the fleet lands ---
FIT=$(sbatch --parsable --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --dependency=afterok:$FLEET \
  --job-name=skilldag-p2-fitters --output=phase2_out/fitters-%j.out \
  --wrap "$PRE; \
    python fit_regmix.py --fleet fleet --out weights_arm2.json; \
    python fit_mixing_law.py --fleet fleet --out weights_arm3.json --t-out mixlaw_t.json; \
    python cluster_tlite.py --mixing-law mixlaw_t.json --k 4 --out clusters.json")
echo "fitters+cluster-> job $FIT  (after fleet)"

# --- 5: arm 5 cluster-level A_ij (10 runs), after clusters exist ---
AIJ5=$(sbatch --parsable $GPU_ARGS --time=12:00:00 \
  --dependency=afterok:$FIT \
  --job-name=skilldag-p2-aij5 --output=phase2_out/aij5-%j.out \
  --wrap "$PRE; python fit_aij.py --out aij_arm5 --cluster-map clusters.json")
echo "arm5 A_ij      -> job $AIJ5  (after clusters, 10 runs)"

cat <<'EOF'

Submitted. Watch with: squeue -u $USER

WHEN THE PROBES FINISH, CHECK THIS BEFORE TRUSTING ANYTHING:

  python -c "import json; a=json.load(open('aij_arm4/aij.json'))['A']; \
  v=[x for r in a.values() for x in r.values()]; \
  print('positive:', sum(1 for x in v if x>0), '/', len(v))"

0 positive entries means A is degenerate and arms 4 and 5 are meaningless. Two
different causes with two different fixes: probes too short (review item 5), or the
token-matching confound (item 13). Do not proceed to Phase 3 on a degenerate matrix.
EOF
