#!/bin/bash
# Phase 0: download + tokenize the Dolma v1.5 domain pools. CPU ONLY, no GPU needed.
#   sbatch farmshare_phase0_prep.sh
# ~87 GB download, ~93 GB on disk, several hours. Resumable: rerun to continue.
#SBATCH --job-name=skilldag-p0-prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=phase0_out/slurm-%j.out
set -e
cd "$(dirname "$0")"
mkdir -p phase0_out

source ../.venv_skilldag/bin/activate

TOKENS_PER_DOMAIN=${TOKENS_PER_DOMAIN:-6000000000}
OUT=${OUT:-dolma_domains}

echo "=== natural-weight estimate (no download) ==="
# If this returns empty sizes, the Accept-Encoding: identity path has regressed
# and every later step is broken. Fail loudly here rather than silently later.
python prep_dolma_domains.py --estimate-only

echo
echo "=== building pools: ${TOKENS_PER_DOMAIN} tokens/domain -> ${OUT}/ ==="
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
echo "PHASE 0 COMPLETE. Pools are memory-mapped at training time, so ${OUT}/ must stay"
echo "on a real filesystem -- object storage cannot be read directly by DomainPools."
echo "Next: sbatch farmshare_phase1_smoke.sh"
