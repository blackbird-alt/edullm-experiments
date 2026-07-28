# eduLLM P1 experiments

Corpus-lever experiments for the eduLLM program (P1, Corpus Foundry track).

- `skill_dag/` — **the Skill-DAG experiment.** Ordering data by dependency
  structure rather than naive easy-to-hard. Does pre-training with adaptive domain
  weights need less compute to reach a given validation loss than fixed weights?
  Five arms (natural, RegMix, Data Mixing Laws, Skill-It A_ij, A_ij + T-LITE
  clustering) over Dolma v1.5. Base: OLMo-1B-hf @ `step1000-tokens4B`. 15 runs.
  **Not yet started** — no data downloaded, no GPU used; 17 review questions open.
  See `skill_dag/PLAN.md`.
- `base_370m/` — from-scratch training pipeline for a shared ~370M base
  (fallback; current plan uses allenai/OLMo-Ladder-760M-0.5xC, revision pinned
  in the runbook).
- `archive/skill_dag_arithmetic_ordering/` — the first Skill-DAG attempt, testing
  prerequisite *ordering* on synthetic arithmetic. Did not work: the model learned
  the number format rather than the arithmetic, then memorised the stream without
  extractable knowledge, and the hard skills floored below the mastery threshold.
  All three are consequences of the synthetic-arithmetic setup. Kept for reference;
  do not build on it. Post-mortem in `skill_dag/PLAN.md`.
