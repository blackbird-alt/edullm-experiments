# Preregistration — P1 Skill-DAG / Prerequisite Sequencing

Status: DRAFT for P8 registry. File before any main-run GPU spend.
Dataset: `skill_dag_v1` (frozen, SEED=20260723). Schedules: `schedules/` (frozen, verified).

## Question
At ~1B scale, does training in prerequisite (topological) order reach mastery in fewer
tokens than a random shuffle of the exact same records, when content, counts, form, and
proportions are held identical and only order changes?

## Hypotheses
- Null: topological order requires the same or more tokens to reach mastery as random order.
- Alternative: topological order reaches mastery in fewer tokens (E_tokens > 1).

## Arms (only order differs)
1. Random shuffle — 3 runs (seeds 201, 202, 203)
2. Topological — 3 runs (distinct valid linearizations, seeds 101, 102, 103; all records
   of every prerequisite node precede any record of the dependent node; programmatically
   verified against dag.json edges)

## Held identical across all runs
Record set (56,200 train records, `skill_dag_v1`), per-skill counts, bare problem->answer
format, base checkpoint (the team's frozen shared base), token budget, sequence length,
batch size, optimizer, constant-LR-with-warmup schedule (no decay — decay would down-weight
late-stream data and confound order; Luo et al. 2025), checkpoint cadence, eval set
(heldout.jsonl, disjoint by exact string for algorithmic skills), eval protocol (greedy,
exact match), analysis code (analyze.py, committed before runs).

## Primary metric
E_tokens = mean tokens-to-mastery(random) / mean tokens-to-mastery(topological),
per skill, aggregated across skills by geometric mean. Mastery threshold: 90% held-out
accuracy, linear interpolation between checkpoints. E_tokens > 1 means sequencing is
more token-efficient. Skills where any run never reaches mastery within budget are
reported as censored and excluded from the ratio (and reported).

## Secondary metrics
Per-skill mastery curves; final accuracy per skill per arm; prerequisite-skill floor.

## Decision rule (committed in analyze.py)
- KEEP sequencing if the 95% bootstrap CI lower bound on E_tokens > 1.05 AND topological
  final accuracy does not trail random by more than 2pp on any skill (floor guard).
- KILL if the CI upper bound <= 1.00, or the floor guard fails.
- Otherwise INCONCLUSIVE (reported as such; no post-hoc threshold moving).
- Bootstrap: 1,000 iterations, resampling runs (within arm) and held-out items (within
  skill) jointly; analyzer verified on synthetic effect (KEEP) and null (INCONCLUSIVE) data.

## Token budget
Set after the pilot (pilot_calibrate.py + a short smoke run): the budget is the smallest
round number at which the random arm reaches 90% on all nine skills, plus 25% margin.
Budget is fixed before the six main runs launch and recorded in run_config.json.
If any skill floors at bare-answer format in the smoke run, its difficulty ranges are
retuned and the dataset refrozen as v2 BEFORE main runs; v1 results are then void.

## Known limitations (declared now)
- Synthetic single-domain arithmetic; establishes the order effect at ~1B, not transfer
  to real curriculum text (Stage 2, separate prereg).
- WORD/EXPR use 5 templates each (randomized fillers); template pattern-matching is
  possible and named in the writeup.
- Fact skills (A, M) are recall tasks: train == eval by design.
- 3 runs per arm bounds run-level bootstrap resolution; item-level resampling supplies
  most of the CI width.

## Compute (estimate, recorded against actuals)
6 runs x (budget tokens) at 370M-1B on cluster GPUs; est. 20-100 H100-hours total
depending on base size and pilot-set budget. Proxy/pilot compute is logged and reported.
