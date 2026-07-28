# Skill-DAG / Prerequisite Sequencing experiment (P1)

Does prerequisite-ordered training beat random order at equal everything-else?
Full design + decision rule: [PREREG.md](PREREG.md). Dataset docs:
[skill_dag_dataset/README.md](skill_dag_dataset/README.md).

## Pipeline (run in this order)

| step | script | where | what |
|---|---|---|---|
| 0 | `build_skill_dag_dataset.py` | CPU (done) | frozen dataset `skill_dag_v1` + DAG |
| 1 | `build_schedules.py` | CPU (done) | 3 topological + 3 random order files, constraint-verified |
| 2 | `pilot_calibrate.py --model <base>` | GPU | per-skill cold accuracy; flags floor/ceiling skills |
| 3 | short smoke run of `train_cpt.py` (~200M tokens, one schedule) | GPU | confirms floored skills lift; sets token budget |
| — | freeze budget, file PREREG.md with P8 | — | REQUIRED before main runs |
| 4 | `train_cpt.py --model <base> --schedule schedules/<name>.idx --token-budget <B> --out runs/<name>` ×6 | GPU | the six main runs (independent; run in parallel) |
| 5 | `eval_mastery.py --run runs/<name>` ×6 | GPU (cheap) | per-checkpoint mastery curves |
| 6 | `analyze.py --topo runs/topo_* --random runs/random_*` | CPU | E_tokens + CI + KEEP/KILL/INCONCLUSIVE |

## Status
- [x] Dataset frozen (verified: 0 answer errors in 3k sample, 0 train/test leakage, no label skew)
- [x] Schedules frozen (topological constraint programmatically verified)
- [x] Analyzer verified on synthetic effect data (KEEP) and null data (INCONCLUSIVE)
- [ ] Pilot on the shared base (needs GPU + base checkpoint decision finalized)
- [ ] Token budget set -> prereg filed with P8
- [ ] Main runs -> eval -> verdict

## Notes for whoever runs the GPU steps
- Constant LR + warmup is deliberate (no cosine): decay would down-weight late-stream
  data and confound the order manipulation. Do not "fix" it.
- `train_cpt.py` never shuffles; the schedule file IS the experiment.
- All six runs must use the same base, budget, and hyperparameters; only `--schedule` differs.
