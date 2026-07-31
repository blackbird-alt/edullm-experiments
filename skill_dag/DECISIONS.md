# Decisions needed before launch

Everything blocking the first real GPU spend, in one place. Nothing here needs a code
change — each item is a value to pick and record.

Two of them (A and B) are new and operational. The rest are the scientific questions that
were already open. Item C is the one that can quietly invalidate the whole experiment.

---

## A. Hardware — settled, just needs three values

The 4 H100s on the group partition. Nothing else comes close: a whole 27-hour run fits in
one job, so there is no chunking, no preemption and no queue, which puts the main phase at
**~4.3 days** (~3.3 with gradient checkpointing off). The public-partition fallbacks are
~4 days on a preemptible H200, ~13 on the L40S ORCD gives out by default, and ~27 if
you are stuck with 6-hour jobs.

**What we need from you**, all from `scontrol show partition <name>`:

1. the partition name
2. its real time limit
3. how many of the 4 H100s are actually yours rather than shared with the group

```bash
PARTITION=pi_yourgroup WALL=7-00:00:00 CONC=4 bash orcd_phase3_main.sh
```

`CONC` is how many runs go at once — lower it if the cards are shared. The launcher
derives its chunk length from `WALL` and falls back to chunk-and-resubmit if a run cannot
finish in one job, so a wrong guess costs time rather than correctness.

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
