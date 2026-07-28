# Preregistration — Skill-DAG

Status: **DRAFT. File before any main-run GPU spend.** Values marked **[SET BEFORE LAUNCH]**
are choices the source doc does not specify; they must be fixed here, not after seeing data.

Supersedes the archived prerequisite-ordering attempt on synthetic arithmetic
(`archive/skill_dag_arithmetic_ordering/PREREG.md`), which failed for reasons specific to
that setup — see "Previous attempt" in `PLAN.md`.

## Question and hypothesis

Verbatim from the proposal doc:

> **Null hypothesis:** Pre-training with a dataset with adaptive domain weights requires the
> same amount of compute to achieve the same validation loss compared to pre-training with a
> dataset with fixed domain weights.
>
> **Alternative hypothesis:** Pre-training with a dataset with adaptive domain weights
> requires less compute to achieve the same validation loss compared to pre-training with a
> dataset with fixed domain weights.

The test is one-sided, so the null is operationalised as **"adaptive requires the same or
more compute"** — the doc's "the same amount" plus the region the alternative excludes.
No other reading is used anywhere in the analysis.

The dependency structure is Skill-It's A_ij adjacency matrix — how much training on domain
*i* improves domain *j* — estimated by probing every pair and then used to set weights.

## Primary metric

**Tokens to reach a given held-out validation loss**, swept across a range of
near-convergence loss targets rather than one cutoff. A single cutoff would let the
conclusion depend on where the cutoff was placed.

A target counts as reached only when validation loss is at or below it for
**2 consecutive evaluations** (`--consecutive 2`). First-crossing detection would let one
lucky evaluation set the value, biasing whichever arm is noisier. Committed in `analyze.py`
before any runs.

Target grid: from the tightest loss every run can sustain-cross up to the loosest still
informative. Computed by `analyze.py:target_grid`, not chosen by hand.

Secondary: final validation loss per arm at fixed budget; per-domain loss curves; realised
weight trajectories for the adaptive arms.

## Arms — all five from the source doc

| Arm | Weights from | Schedule |
|---|---|---|
| 1 | natural (Dolma's measured mix) | fixed |
| 2 | part-way runs → power-law extrapolation → LightGBM → minimise | fixed |
| 3 | same fleet, data-mixing law instead of LightGBM | fixed |
| 4 | Skill-It A_ij, full pairwise probing @~100M | adaptive, **5** updates |
| 5 | arm 4 + T-LITE k-means clustering, probing at cluster level | adaptive, 5 updates |

Arms 2 and 3 share one proxy fleet (the doc: arm 3 is "same as (2), but with the data
mixing law"), so 2-vs-3 isolates regressor family.

**Baseline for all comparisons:** arm 1.
**Primary contrast:** arms 4 and 5 (adaptive) against arms 2 and 3 (fixed).

## Held identical across arms

Base checkpoint (`allenai/OLMo-1B-hf` @ `step1000-tokens4B`), token budget, sequence
length, batch size, optimiser, LR schedule, evaluation cadence, held-out validation slices
(reserved before any sampling; no arm can train on them), token pool, and analysis code
(`analyze.py`, committed before runs). Only the domain weights and whether they update
differ.

## Decision rule

- **Adaptive wins** if, for arms 4/5 vs arms 2/3, the 95% bootstrap CI upper bound on the
  compute ratio is below 1.0 at a majority of swept targets.
- **Null retained** if the CI includes 1.0 at a majority of targets.
- **Adaptive loses** if the CI lower bound exceeds 1.0 at a majority of targets.

Bootstrap resamples seeds within each arm, 2000 iterations, seed 7.

## Non-inferiority margin — **[SET BEFORE LAUNCH]**

The doc requires arm 5 and the follow-up to be "not statistically significantly worse"
than full pairwise Skill-It. That is a claim of *sameness*, which is untestable without a
stated margin — there is no unmargined equivalence test.

**Proposed margin: 5% of compute** (`--margin 0.05`). Arm 5 is non-inferior to arm 4 if
the CI upper bound on their compute ratio is ≤ 1.05. Change this number here, before
launch, or not at all.

## Compute accounting

**Fitting compute is reported separately from training compute, never pooled.** Arms 2–5
each spend GPU time choosing weights before training starts (arm 4: 45 probe runs; arm 5:
10; arms 2/3: one shared 96-run fleet). The doc's success bar for arm 5 and for the
derivative follow-up compares *fitting* cost across methods, so pooling would erase the
quantity under test. Each fitter writes its own cost record; `analyze.py --fitting-costs`
reports them in a separate table.

**Arm 5's fitting cost is reported two ways, both preregistered here.** Arm 5's clustering
signatures come from arm 3's mixing-law `t` matrix, which requires the 96-run fleet. So:

- **Marginal cost** — 10 cluster probes only (~3 GPU-h), correct if arm 3 is being run
  anyway, as it is here.
- **Standalone cost** — 10 cluster probes plus the fleet that produced the signatures
  (~23 GPU-h), correct for anyone adopting arm 5 on its own.

Against arm 4's ~12 GPU-h, arm 5 wins on the first and loses on the second. The doc's
criterion ("less probing/model-fitting compute than pairwise Skill-It") does not say which
applies, so **both are reported and neither is chosen after seeing results.** The headline
claim uses the standalone figure, because the criterion is about the method rather than
about this experiment's bundling.

## Scale — **[SET BEFORE LAUNCH]** where marked

| | value | source |
|---|---|---|
| Main model | OLMo-1B from `step1000-tokens4B` | doc |
| Proxy models | 50M / 75M / 100M | doc ("various 50-100M"; ~100M for arm 4) |
| Tokens per main run | **~2B** **[SET]** | Skill-It, the only continual-PT peer, used 1B and 3B |
| Seeds per arm | **3** **[SET]** | DataDecide ran 3 seeds at its own 1B target to quantify run-to-run variance, finding SDs up to 2 accuracy points; <3 gives weakly identified intervals |
| Reweighting rounds | 5 | doc |
| Proxy fleet | 32 Dirichlet mixtures × 3 sizes **[SET]** | doc gives no count |
| Probe run length | 200M tokens **[SET]** | doc gives no length |
| Clusters for arm 5 | K=4 **[SET]** | doc gives no K |
| Dataset | Dolma v1.5, k=9 domains, 6B/domain cap **[SET]** | doc names no dataset; follows from "OLMo-1B" |

## Declared limitations

- **Fixed-vs-adaptive is partly confounded with method.** Arms 2/3 are fixed and 4/5
  adaptive, but they also use different weight-selection methods, so "adaptive wins" is
  partly "Skill-It beats regression." Two of each makes it 2-vs-2 rather than 1-vs-1.
  Fully isolating it would need a sixth arm (Skill-It weights held fixed). Reported, not
  fixed — this is the doc's design and it is not wrong, only limited on this axis.
- **The branch point is mid-warmup.** OLMo-1B warms up over 2000 steps, so `step1000` sits
  about halfway up the LR ramp rather than in steady state. It is the earliest published
  checkpoint and the doc asks for "very early."
- **Batch size is 32× smaller than OLMo-1B's original** (64 sequences vs 2048), which
  changes the gradient noise scale relative to the original run.
- **Constant LR** rather than the original's decay to 4e-5, chosen so a decaying LR does
  not down-weight late-arriving data — which in adaptive mode is exactly the data the
  reweighting selected.
- **wiki and books cannot be enlarged.** Dolma v1.5 holds only ~3.6B and ~4.3B tokens of
  them and the plan already takes 100%. Under heavy upweighting they are the first to run
  dry; the trainer logs pool exhaustion so this surfaces in run 1.
- **Validation loss is measured on held-out slices of the same domains**, so it measures
  in-distribution fit, not transfer to an external benchmark. The source doc names this as
  a weak point in the literature — labs care about benchmark scores at equal compute — so
  a loss-only result speaks to the hypothesis as stated but not to the doc's stated gap.
  If an OLMES-style benchmark pass is added to the 15 main runs (inference only, negligible
  compute), it is **secondary and exploratory**, declared here so it cannot be promoted to
  the primary outcome after the fact.
- **The A_ij estimator's token-matching convention materially changes the result** and is
  not settled at the time of writing (plan review item 13). If pairwise probes hold total
  tokens fixed rather than j's tokens, A_ij is biased negative, the Skill-It clip zeroes it,
  and arms 4/5 reproduce their initial weights — retaining the null by construction. The
  convention used must be fixed before probing, and the positive-entry count of A is
  reported as a diagnostic either way.
- **Extrapolation is guarded, not exact.** Power-law fits fall back to the last observed
  value when r² < 0.5 or the implied gain exceeds 40%; weight search is restricted to the
  per-domain range the fleet actually visited. Both guards are on by default and their
  activation counts are recorded — a fitter that mostly fell back is reported as such.
- **The proxies are undertrained relative to any published precedent.** At 200M tokens the
  50–100M proxies sit at a token-to-parameter ratio of 2–3, against Chinchilla-optimal 20
  and the ratio of 100 used at every rung of DataDecide's ladder. If mixture differences do
  not separate at that ratio, arms 2–5 are fitting noise. Reported as a limitation if the
  sizes are not changed before launch; `analyze.py` reports guard-activation counts, which
  is the observable symptom.
- **The multi-size fleet may not earn its cost.** DataDecide found that none of 8
  scaling-law variants beat single-scale ranking on the compute-to-decision-accuracy
  frontier. Arms 2/3 pay 3× for the size axis. It is retained because Data Mixing Laws
  (arm 3's method) requires nested size and token laws, but the axis is not independently
  justified by decision-accuracy evidence.

## Registered conditional follow-up

Only if the null is rejected: estimate A_ij from the derivative of the data-mixing function
with respect to one domain weight, and probe only the highest estimated A_ij values.

The source doc states **three** requirements, all of which must hold:

(a) less compute in the probing and model-fitting stage than full pairwise Skill-It
probing, training compute excluded;
(b) not significantly worse pretraining compute to reach the same validation loss **at any
near-convergence target** — note "any", not "a majority", which is stricter than the
decision rule used for the main contrast above; and
(c) **not significantly worse convergence validation loss.** This is a separate bar on the
final loss itself, not on compute-to-loss, and an earlier draft of this prereg omitted it.
A method that reaches every intermediate target efficiently but plateaus higher fails.

Both (b) and (c) are judged against the same 5% margin. Registered now so this cannot
become a post-hoc analysis.

## Analysis code

`analyze.py`, committed before any main run. Sustained-crossing detection, target grid,
bootstrap, and non-inferiority test are all in it; no post-hoc threshold movement.
