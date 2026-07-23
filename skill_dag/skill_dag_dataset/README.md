# Skill-DAG Stage-1 dataset (`skill_dag_v1`)

Frozen, reproducible procedural-arithmetic dataset for the P1 **prerequisite-sequencing**
experiment (topological order vs. random shuffle of the *same* records). The graph is the
schema: every record is tagged to a node in `dag.json`.

Regenerate byte-identical: `python ../build_skill_dag_dataset.py` (SEED=20260723).

## Files
- `train.jsonl` — 56,200 records (train pool)
- `heldout.jsonl` — 7,200 records (mastery eval pool)
- `dag.json` — the 9-node prerequisite graph (nodes, edges, tiers)
- `report.json` — build stats + integrity checks

Record schema: `{"skill","prompt","answer","text","split"}`. Format is **bare `problem = answer`**,
no worked steps — deliberately, so the only thing the experiment varies is training *order*.

## The graph
Tier 0 (fact roots): A (single-digit +), M (single-digit ×)
Tier 1: ADD, SUB (←A), MUL (←M, ADD)
Tier 2: DIV (←MUL, SUB), FRAC (←MUL, ADD)
Tier 3: EXPR (←ADD,SUB,MUL,DIV), WORD (←ADD,SUB,MUL)
Every edge is load-bearing: the dependent skill uses the prerequisite as a literal sub-step.

## Integrity safeguards (enforced, checked in report.json)
- **Correctness:** answers computed from operands (correct by construction); an independent
  verifier re-checked 3,000 sampled records → 0 incorrect.
- **No leakage:** algorithmic skills' held-out pool is disjoint from train by exact problem
  string (overlap = 0, asserted). Fact skills (A, M) are recall tasks over all 100 facts, so
  train == eval **by design** (flagged, not a leak).
- **Unbiased answers:** most-frequent answer's share is <2.5% for every skill (no shortcut label).
- **Balanced magnitude:** operands sampled by drawing digit-length uniformly, then value
  uniformly — no skew toward large numbers.
- **Fixed format** across all skills, so format is not a confound with the order manipulation.

## Known limitations (honest, for the writeup)
- **Template bias (WORD, EXPR):** 5 templates each. Numbers/names/objects are randomized, but a
  model could pattern-match templates rather than reason. Acceptable for a Stage-1 order test;
  name it in results. WORD answers are correct by construction but not independently re-verified
  by the sampler.
- **Bounded ranges = no extrapolation test.** Skills are mastered within the trained digit ranges;
  this does not test generalization to larger operands. Add an out-of-range transfer set if wanted.
- **Fact skills = recall, not generalization** (train == eval), which is the correct notion of
  mastery for arithmetic facts but is not a held-out test.
- **Difficulty is not yet calibrated to the base model.** The current ranges (e.g., 2–3 digit ×
  2 digit for MUL) may be too hard for a small model to learn from *bare answers* — prior work
  (Havrilla & Iyer; Deng et al.) shows answer-only multi-digit multiplication floors at small
  scale. The **pilot must tune the digit ranges down per skill** until each skill's bare-answer
  accuracy lands mid-range (not 0, not 100) on the chosen base. The generator exposes those knobs.
- **Synthetic, single-domain (arithmetic).** A win here is Stage-1 evidence that prerequisite order
  helps at ~1B; it does not establish transfer to real educational content (that is Stage 2).
