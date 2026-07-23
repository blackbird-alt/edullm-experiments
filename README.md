# eduLLM P1 experiments

Corpus-lever experiments for the eduLLM program (P1, Corpus Foundry track).

- `skill_dag/` — prerequisite-sequencing experiment: frozen dataset + DAG,
  verified topological/random schedules, CPT/eval/analysis pipeline, prereg,
  FarmShare runbook. See `skill_dag/RUNBOOK.md`.
- `base_370m/` — from-scratch training pipeline for a shared ~370M base
  (fallback; current plan uses allenai/OLMo-Ladder-760M-0.5xC, revision pinned
  in the runbook).
