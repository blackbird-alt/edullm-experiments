#!/bin/bash
# Phase 0: download + tokenize the Dolma v1.5 domain pools. CPU ONLY, no GPU needed.
#   sbatch orcd_phase0_prep.sh
#
# ~87 GB download, ~93 GB on disk, several hours. Resumable per shard: rerun to continue.
#
# mit_normal caps jobs at 12 hours, which may not be enough for the full download plus
# tokenization on a first pass. That is fine -- completed shards are skipped on restart,
# so resubmitting picks up where it stopped. Watch the first run to see how far it gets.
#SBATCH --job-name=skilldag-p0-prep
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=phase0_out/slurm-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p phase0_out

source "$HOME/orcd/pool/venv_skilldag/bin/activate"

# Pools must live somewhere with room and real filesystem semantics -- they are
# memory-mapped at training time. Scratch is 1 TB and flash-backed; home is only 200 GB
# and would be tight once checkpoints land next to it.
OUT=${OUT:-$HOME/orcd/scratch/skilldag/dolma_domains}
TOKENS_PER_DOMAIN=${TOKENS_PER_DOMAIN:-6000000000}
mkdir -p "$(dirname "$OUT")"

echo "=== natural-weight estimate (no download) ==="
# If this returns empty sizes, the Accept-Encoding: identity path has regressed
# and every later step is broken. Fail loudly here rather than silently later.
python prep_dolma_domains.py --estimate-only

echo
echo "=== building pools: ${TOKENS_PER_DOMAIN} tokens/domain -> ${OUT} ==="
echo "wiki (~3.6B) and books (~4.3B) are smaller than the cap -- that is all of them"
echo "that exists in Dolma v1.5. They will stop short; this is expected, not an error."
echo

python prep_dolma_domains.py \
  --tokens-per-domain "${TOKENS_PER_DOMAIN}" \
  --out "${OUT}" \
  --procs "${SLURM_CPUS_PER_TASK:-8}"

echo
echo "=== result ==="
ls -lh "${OUT}"/
cat "${OUT}"/manifest.json

echo
echo "PHASE 0 COMPLETE."
echo "Pools are memory-mapped at training time, so ${OUT} must stay on a real"
echo "filesystem. Scratch is not backed up and is purged after 6 months idle."
echo "Export DATA=${OUT} for the later phases, then: sbatch orcd_phase1_smoke.sh"
