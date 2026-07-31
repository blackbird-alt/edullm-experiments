# Decisions needed before launch

Everything blocking the first real GPU spend, in one place. Nothing here needs a code
change — each item is a value to pick and record.

Two of them (A and B) are new and operational. The rest are the scientific questions that
were already open. Item C is the one that can quietly invalidate the whole experiment.

---

## A. Hardware and partition — operational, decide first

**Recommendation: the group partition with the 4 H100s.**

| where | wall clock, main phase | caveats |
|---|---|---|
| **4 owned H100s** | **~4.3 days** (~3.3 without gradient checkpointing) | one job per run, no queue, no preemption |
| `mit_preemptable`, H200 | ~4 days | 4 GPUs, jobs killed at any time, hours of queue |
| `mit_preemptable`, L40S | ~13 days | the ORCD default GPU; 2 chunks per run |
| `mit_normal_gpu`, L40S | ~27 days | 6 h cap and 2 GPUs means ~210 queue waits |

Owned H100s win on every axis: a whole 27-hour run fits in one job, so there is no
chunking, no preemption risk, and the 4.3 days is a real number rather than a floor that
queue time inflates.

```bash
PARTITION=pi_yourgroup WALL=7-00:00:00 CONC=4 GPU=h100 bash orcd_phase3_main.sh
```

**What we need from you:** the partition name, its actual time limit
(`scontrol show partition <name>`), and whether the 4 H100s are exclusively yours or
shared with the group. If shared, lower `CONC`.

The code handles all four rows — it chunks and resubmits when a run cannot fit in one job,
and switches from `-G h100:1` to `--gres=gpu:1` off the public partitions. This decision
only changes how long you wait.

---

## B. Scope, only if compute turns out to be short

At ~4 days on the H100s, **the recommendation is to change nothing.** Listed so the option
is on the record, because a scope cut chosen *after* seeing results is not a scope cut, it
is p-hacking.

| cut | saves | what it costs |
|---|---|---|
| gradient checkpointing off | ~25% | nothing scientific; free, do it if memory allows |
| 2 seeds instead of 3 | 33% | the variance estimate the non-inferiority test rests on. DataDecide ran 3 seeds at this exact scale for this exact reason |
| 1B tokens instead of 2B | 50% | mixture effects get smaller and noisier in shorter runs. Already only ~2 tokens per parameter, far below Chinchilla |

The first is decided by measurement, not judgement: Phase 1 reports whether activations fit
on an 80 GB H100 without checkpointing. They almost certainly do.

Halving the budget is a one-variable change (`BUDGET=1000000000`), and the evaluation
cadence now scales with it automatically so the analysis keeps its eight measurement
points. It is cheap to do and expensive to justify.

---

## C. The A_ij confound — blocks Phase 2, highest severity

Review item 13 in [PLAN.md](PLAN.md). `fit_aij.py` holds *total* tokens fixed across
probes rather than holding *j's* tokens fixed. Every entry of the transfer matrix is
therefore biased negative, and Skill-It's `max(A, 0)` clip then pins the adaptive arms at
whatever weights they started with.

The failure mode is what makes this urgent: arms 4 and 5 would run to completion, cost
their full share of the compute, and produce a clean-looking null result that is an
artifact of the estimator rather than a finding about the hypothesis. You would not be able
to tell from the output that anything was wrong.

`orcd_phase2_fit.sh` refuses to run until this is acknowledged
(`I_HAVE_SETTLED_ITEM_13=yes`). There is also a check to run the moment the probes finish —
if the matrix has zero positive entries, stop, because either this or review item 5
(probes too short) has swallowed the signal, and the two have different fixes.

---

## D. The remaining review questions

17 others in [PLAN.md](PLAN.md) under "Questions for review". They range from design
choices worth a second opinion to things that would need a code change if you disagree.
Worth reading before Phase 2, since several affect the fitting runs.

## E. The preregistration

9 `[SET BEFORE LAUNCH]` values are still blank in [PREREG.md](PREREG.md), including the
token budget and the non-inferiority margin. **The prereg has to be filed and frozen before
Phase 3 starts** — that is the program rule, and it is the whole reason `analyze.py` was
written and committed before any run exists.

Two things from section A and B belong in it: the token budget, and whether the adaptive
arms start from the natural mix or from arm 2's fitted optimum (the source doc says
"optimal domain weight to start" without saying which, so the launcher defaults to
`natural` and exposes `ADAPTIVE_INIT`).

---

## Safe to run right now, regardless

Phase 0 (download and tokenize, CPU) and Phase 1 (30-minute GPU smoke test) are unaffected
by any of the above and cost almost nothing. Phase 1 in particular should happen before any
of these decisions are final, because **nothing in this repo has ever run on a GPU** — it
reports measured throughput, which replaces every estimate in this document, and it is the
cheapest way to find the bugs that only appear when code actually runs.

```bash
sbatch orcd_phase0_prep.sh
sbatch orcd_phase1_smoke.sh
```

Full operational detail in [RUNBOOK.md](RUNBOOK.md).
