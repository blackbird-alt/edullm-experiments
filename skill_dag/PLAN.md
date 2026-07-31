# Skill-DAG

Ordering data by dependency structure, not by naive easy-to-hard. The dependency structure
is **Skill-It's A_ij adjacency matrix** — how much training on domain *i* improves domain
*j* — estimated by probing every pair, then used to set the domain weights.

Arms 4 and 5 are the Skill-DAG arms. Arms 1–3 are the baselines they must beat: natural
weighting, RegMix, and Data Mixing Laws.

Prior evidence for the idea: Skill-It reached with 1B tokens what uniform sampling needed
3B for (~3×); DoReMi, 2.6× fewer steps; RegMix matched DoReMi at 10% of the compute.

## Hypothesis

Stated verbatim from the proposal doc:

> **Null hypothesis:** Pre-training with a dataset with adaptive domain weights requires the
> same amount of compute to achieve the same validation loss compared to pre-training with a
> dataset with fixed domain weights.
>
> **Alternative hypothesis:** Pre-training with a dataset with adaptive domain weights
> requires less compute to achieve the same validation loss compared to pre-training with a
> dataset with fixed domain weights.

DV = **validation loss**. IV = weighting method and schedule. The test is one-sided, so the
null is operationalised as "the same or more compute" in `PREREG.md`.

## Previous attempt — why this design is different

An earlier Skill-DAG attempt tested prerequisite *ordering* on synthetic arithmetic
(topological vs random order over a hand-built skill DAG, 1B scale). It is archived at
`archive/skill_dag_arithmetic_ordering/`. It did not work, and the failures were all
consequences of the synthetic-arithmetic setup rather than of the dependency-graph idea:

- **v1 learned the format, not the arithmetic** — BPE chunked multi-digit numbers, so the
model matched surface patterns. Fixed by respacing digits; dataset refrozen as v2.
- **An eval bug produced flat curves** — a trailing prompt space that BPE attached to the
answer token, which also explained an apparent pilot regression.
- **v2 memorised the stream verbatim** — full-loss packed CLM hit loss 0.011 with ~0 bare
extraction: the Allen-Zhu memorised-but-not-extractable failure, reproduced. Fixed by
per-example answer-masked loss in v3.
- **Difficulty was bimodal and the hard skills floored** — after all that, MUL/DIV sat at
~0.20 accuracy and were expected to be censored outright, and the mastery threshold had to
be cut from 0.90 to 0.80. The measurement could not reach the skills that mattered.

Its own prereg named the root cause as a declared limitation: "synthetic single-domain
arithmetic ... not transfer to real curriculum text."

**This design removes that whole failure class.** It trains on real text (Dolma v1.5) and
measures validation loss rather than exact-match mastery on generated answers. There is no
synthetic format to memorise instead of learning, no extraction step that can silently
return zero, and no per-skill accuracy threshold that hard skills can floor beneath. Loss is
continuous and always measurable, so no arm can produce an unreadable result — which is also
what DataDecide found, that continuous likelihood metrics carry signal at small scale where
discrete accuracy does not.

The risks that remain are different ones, and they are listed as review questions below —
above all item 13, which is the one way this design could still manufacture a null.

## Questions for review

Ranked by how much compute is wasted if the assumption is wrong. Items 2–4 are
discrepancies against OLMo-1B's *actual* training config
(`configs/official/OLMo-1B.yaml`, v0.2.5). Items 13–18 came from checking this plan against
the original proposal doc; **item 13 is the single highest-severity item on the list** and
should be read first — it can null the experiment by construction.

1. **Is ~2B tokens per run enough for the arms to separate?** If not, every run produces
  flat curves and the whole experiment is wasted. The basis is stronger than an earlier draft
   of this plan claimed: the proposal doc's figure is that **Skill-It at 1B tokens matched
   uniform sampling at 3B**, so the effect was already measurable at a 1B budget — not merely
   that 1B and 3B were two settings someone tried. A 2B budget sits above the point where the
   effect has been observed. That is reassurance about the *budget*; it says nothing about
   whether the effect transfers to continual pretraining of a 1B model (item 15).
2. **Batch size is 32× smaller than OLMo-1B's.** Original: `global_train_batch_size: 2048`
  sequences × 2048 tokens = **4.19M tokens/step**. This plan: 8 × grad-accum 8 = 64
   sequences = **131k tokens/step**. Dropping batch size by 32× when resuming mid-training
   changes the gradient noise scale and may destabilize. Should the batch be matched, and
   if so does that change the memory/parallelism plan?
3. `step1000-tokens4B` **is *before* warmup finished.** Original `t_warmup: 2000`, so at
  step 1000 the model was about halfway up the LR ramp (roughly 2e-4 of a 4e-4 peak), not in steady
   state. Is that a sound branch point, or should this start from a post-warmup checkpoint
   (e.g. `step5000-tokens20B`) even though it is less "very early"?
4. **Constant LR vs the original schedule.** Original: peak 4e-4, `alpha_f: 0.1` (decays to
  4e-5). This plan uses constant 3e-4 with 200 warmup steps. Constant was chosen so a
   decaying LR wouldn't down-weight late-arriving data — which in adaptive mode is exactly
   the data the reweighting selected, confounding the comparison. Is that trade right, and
   is 3e-4 the right level to resume at?
5. **The proxies are 5–10× undertrained, and DataDecide says that is the wrong end to
  economise on.** Probes and fleet runs are fresh random-init models trained on 200M tokens.
   At the 100M preset (hidden 640, 10 layers → ~66M non-embedding, ~98M total) that is a
   token-to-parameter ratio of about **2–3**, against Chinchilla-optimal 20 and the ratio of
   **100** used for every rung of the OLMo model ladder in DataDecide. DataDecide's smallest
   model — 4M parameters — saw 0.4B tokens, twice what these 100M proxies get. So the fleet
   sits entirely outside the regime anyone has shown mixture signal to exist in, and if the
   proxies have not yet learned enough language, A_ij is noise and arms 4 and 5 are
   meaningless — 55 probe runs wasted plus six main runs built on a garbage matrix.
   Two ways out, and they cost very differently:
   (a) **lengthen the runs** to ~2B tokens to reach ratio 20, which is 10× the fitting
   compute — arm 4 goes 12 → ~125 GPU-h and the fleet 20 → ~200, roughly doubling the whole
   experiment; or
   (b) **shrink the proxies** to ~10–20M parameters, which reaches ratio 10–20 at the
   current 200M tokens for **no extra cost**. RegMix fitted its regressor on 1M-parameter
   models, and DataDecide's ladder includes a 10M rung, so this is well inside precedent.
   The obstacle is only that the source doc says "various 50-100M".
   **Note that the doc argues against itself here.** Its own evidence section describes
   RegMix as training **512 models of 1M parameters**, then its arm-2 text calls for
   "various 50-100M sized models". This plan followed the arm text: 32 mixtures at 50–100M.
   That is 16× fewer mixtures than RegMix used and 50–100× larger models — the opposite
   trade on both axes, from a doc that cites RegMix as the method being reproduced. For a
   9-dimensional simplex, 32 sample points is thin regardless. Going smaller and more
   numerous is now supported by RegMix, by DataDecide's ladder, and by the ratio argument
   above.
6. **Mirror-descent step size** `eta = 0.5` **is arbitrary.** Too small and the adaptive arms
  barely move (collapsing into arm 1); too large and weights swing wildly. No principled
   basis for the value.
7. **The fixed-vs-adaptive comparison is partly confounded with method** (see Arms below).
  Worth a sixth arm, or accept and report?
8. **Non-inferiority margin is unset.** The doc requires "not statistically significantly
  worse" but names no threshold; it must be fixed before running.
9. **Seeds now read disjoint data** — fixed since first draft; each replicate starts in its
  own pool region rather than all at position 0. Remaining question: at a realistic 40% peak
   weight the seven 6B domains have 2.5× headroom but **wiki has only 1.5×**, and at an 80%
   peak wiki goes underwater and its seeds overlap. Is 1.5× acceptable, or should runs be
   shortened to 1.5B tokens to widen it? (wiki cannot grow — see item 10.)
10. **wiki and books cannot be made bigger.** Dolma v1.5 holds only about 3.6B and 4.3B tokens
  of them, and the plan already takes 100%. Under heavy upweighting those two are the first
   to run dry. Is capping the pool at 6B/domain the right call, or should the small domains
   be merged, dropped, or handled some other way?

11. **Optimiser state is bf16, not fp32.** The model is loaded in bf16 and AdamW is built
   directly on those parameters, so its moment estimates are bf16 too. Common practice for
   long pretraining is fp32 master weights / optimiser state, because bf16 has ~8 bits of
   mantissa and small updates can be lost to rounding. This affects all five arms equally so
   it does not bias the comparison, but it could degrade every run. Should the runs use mixed
   precision with an fp32 master copy instead?
12. **One learning rate across all three proxy sizes confounds the size trend arm 3 depends
  on.** `fit_proxy_fleet.py` trains 50M, 75M and 100M at the same constant `--lr 3e-4` with
   no warmup, from random init. Optimal LR falls as model size rises, so a single value is
   mistuned by a *different amount* at each size — and `extrapolate.py:extrapolate_to_size`
   fits loss against size and extrapolates that trend to 1B. The trend therefore absorbs the
   mistuning gradient along with the real size effect. DataDecide states the requirement
   directly: hyperparameters must be set per scale "to avoid confounding in performance
   differences that are simply due to suboptimal hyperparameters," and its ladder sets both
   LR and batch size per rung (60M: 5.8e-3 / 96 seq; 90M: 4.9e-3 / 160; 150M: 4.2e-3 / 192).
   This plan uses 3e-4 and 64 sequences at every size — roughly 16–19× below the ladder's LR
   for models this small, and identical across sizes. Adopt the ladder's per-size LR and
   batch, or drop to a single proxy size (see below) so there is no trend to corrupt.

13. **The A_ij probe holds total tokens fixed instead of j's tokens, which biases every
  entry negative and can silently freeze the adaptive arms at their starting weights.** The
   proposal doc
   defines the pairwise estimator as "training on datasets i and j vs only j and then
   evaluating on skill j." That phrasing does not say what is held constant, and `fit_aij.py`
   holds the *total* budget constant: the single run is 200M tokens of 100% j, and the pair
   run is 200M tokens of 50% i / 50% j — so **j gets half as many tokens in the pair run**.
   `A[i][j] = alone[j] - pair_loss_j` therefore measures "is half of i better than the other
   half of j," not "does i help j." Halving j's own in-domain data raises j's loss, so the
   estimator carries a large constant negative bias against every pair. `skillit_update`
   then applies `max(A_ij, 0)`. If the bias dominates, every entry clips to zero, every
   `exp(eta * 0) = 1`, and the weights never move: **arms 4 and 5 reproduce their starting
   weights exactly and the null is retained by construction rather than by evidence.** This
   is the worst failure mode in the plan because it looks like a clean result.
   The fix is to hold *j's* token count fixed rather than the total — single = 200M of j,
   pair = 200M of j **plus** 200M of i — which makes A_ij the causal "adding i on top of j"
   quantity the doc describes. Cost goes from 45 × 200M = 9B tokens to 9 × 200M + 36 × 400M
   = 16.2B, so arm 4 moves from ~12 to ~22 GPU-h. That is cheap next to discovering the
   matrix was meaningless after the main runs. `fit_aij.py` already prints a positive-entry
   count, but it attributes an all-zero A to short probes (item 5); under this confound the
   same symptom has a different cause and a different fix, so the diagnostic needs to
   distinguish them.
14. **Arm 5's "cheaper probing" claim depends on how the fleet is charged, and as written it
  may fail its own success criterion.** The doc's bar for arm 5 is "less probing/model-fitting
   compute than pairwise Skill-It." Arm 5 runs 10 cluster probes (~3 GPU-h) against arm 4's
   45 (~12), which looks like a clear win — but arm 5's clustering signatures come from the
   arm-3 mixing-law `t` matrix, which requires the 96-run fleet (~20 GPU-h). The plan calls
   that "already computed, free," which is only true because arm 3 is in the same experiment.
   Standalone, arm 5 costs ~23 GPU-h against arm 4's ~12 and is **more** expensive. Both
   numbers should be preregistered and reported — marginal cost given arm 3, and standalone
   cost — because which one is used decides whether arm 5 passes. The doc's other offered
   signature, a row of A, is worse: it needs arm 4's full 45 probes first.
15. **The doc's stated effect size is 30–35%, not 3×, and nothing checks that 3 seeds can
  detect it.** The doc's evidence summary says these methods give "a moderate reduction in
   compute (~30–35%) and a few pp improvement in benchmark scores," while its headline cites
   Skill-It at ~3× and DoReMi at 2.6×. The conservative figure is the one to power against,
   and there is no power analysis anywhere. With 3 seeds per arm the bootstrap CI is built
   from 3 values; whether a 30% effect clears it depends entirely on between-seed variance in
   tokens-to-target, which is unmeasured. Arm 1's three seeds in the pilot would give that
   estimate cheaply, and it should be checked *before* committing to 3 seeds.
16. **The DV is validation loss only, but the doc's weak-evidence section is about
  benchmarks.** The doc says labs care about "achieving exactly the same benchmark scores
   with slightly reduced compute or a few pp higher on benchmark scores with equal compute,"
   and names this as a place the literature is weak. This plan measures held-out loss on the
   same nine domains and declares external transfer out of scope. Adding an OLMES-style
   benchmark pass at the end of each of the 15 main runs is **inference only** — negligible
   compute against hundreds of GPU-hours of training — and it is the one addition that would speak to the doc's own
   stated gap. DataDecide supplies the task list and shows that character-normalised
   likelihood metrics carry signal at small scale where raw accuracy does not.
   *Now built:* `eval_benchmarks.py` scores nine OLMES tasks by length-normalised
   likelihood and reports each task's chance rate beside the result. It is deliberately
   wired as a secondary measure that feeds nothing in `analyze.py`, so the remaining
   decision is only whether to name it in the prereg as a reported-but-not-decisive
   outcome. Leaving it out of the prereg and running it anyway would make it a post-hoc
   measure, which is the failure mode worth avoiding here.
17. **What the adaptive arms start from is unspecified, and it changes what arms 4/5 test.**
  The doc says arms 4 and 5 "select optimal domain weight to start, and then adjust the
   weights 5 times throughout pre-training according to the skill-it formula." It never says
   what "optimal to start" means. Two readings: start at the **natural** mix, so arms 4/5
   test Skill-It reweighting alone; or start at **arm 2's fitted optimum**, so they test
   Skill-It *on top of* RegMix. The second makes 4-vs-2 a clean test of "does adapting help
   beyond a good fixed choice," but it also means arms 4/5 inherit the whole 96-run fleet
   cost and are no longer independent of arm 2. `orcd_phase3_main.sh` defaults to
   natural and exposes `ADAPTIVE_INIT` for the alternative. This interacts with item 13: if
   A is degenerate the adaptive arms never move, so whichever start is chosen is *also* the
   final answer, and arms 4/5 silently become a duplicate of arm 1 or arm 2.
18. **T-LITE's candidate selection is offered by the doc and is not implemented.** The doc
  says arm 5 may "optionally mirror T-LITE's candidate selection as well: for each skill,
   only probe the top-N candidate prerequisite clusters by the derivative estimate." Nothing
   in `fit_aij.py` does this — arm 5 probes all 10 cluster pairs. It is marked optional in
   the doc, so leaving it out is legitimate, but it should be declared rather than silently
   dropped, since it overlaps with the registered derivative follow-up.

**DataDecide** (arXiv:2504.11393) — read. Four things bear on this plan.

*Against the multi-size fleet.* Its headline result is that **none of 8 scaling-law variants
beat the compute-to-decision-accuracy frontier of simply ranking experiments at one small
size**; ranking at 150M alone gets ~80% of pairwise data-recipe decisions right for a 1B
target. Arms 2 and 3 pay 3× for the size axis (96 runs instead of 32) to do exactly the
multi-scale extrapolation DataDecide found buys nothing. Dropping to one size would cut the
fleet from ~20 to ~7 GPU-h and free budget for more mixtures — and 32 mixtures is thin for a
9-dimensional simplex, where RegMix used 512 runs. **Counter-argument, and it is a real
one:** Data Mixing Laws (arm 3's source method) *requires* nested size and token scaling
laws, so dropping the size axis departs from the method arm 3 is supposed to be testing, and
DataDecide's decision task is ranking 25 discrete corpora rather than minimising a regressor
over a continuous simplex. If the size axis stays, item 12 must be fixed, because arm 3 then
leans on a size trend that a fixed LR corrupts.

*Supporting current choices.* Intermediate checkpoints make decisions as well as
compute-equivalent final checkpoints, which is what `fit_proxy_fleet.py`'s 5-point curves
assume. Continuous metrics beat discrete accuracy at small scale, supporting validation loss
as the DV. And DataDecide ran **3 seeds at its 1B target specifically to quantify run-to-run
variance**, finding standard deviations as high as 2 accuracy points — direct precedent for
this plan's 3 seeds per arm, which until now was justified only as "the floor."

## Setup (verified, not assumed)


| Item          | Value                                                                       | How verified                                                                            |
| ------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Base          | `allenai/OLMo-1B-hf` rev `step1000-tokens4B`                                | refs API: 351 checkpoints; step 1000 of 738020; loads directly, no conversion           |
| Early ckpts   | sub-step-20000 exist **only** on the `-hf` repo                             | native `allenai/OLMo-1B` starts at step20000                                            |
| Tokenizer     | repo's own (GPTNeoX, vocab 50280 — fits uint16)                             | `tokenizer_config.json`                                                                 |
| Corpus        | Dolma **v1_5**                                                              | the corpus OLMo-1B trained on; the doc names no dataset, so this follows from "OLMo-1B" |
| Domains (k=9) | cc_en_head, cc_en_middle, cc_en_tail, c4, stack, reddit, pes2o, books, wiki | manifest `urls/v1_5.txt`                                                                |
| Access        | public, ungated, ODC-BY                                                     | HTTP 200 on all 9                                                                       |


Gotchas: filenames differ per domain (`en_simple_wiki_v0-*`, `pes2o_v2-*`), so URLs must
come from the manifest, never constructed. v1.5 `wiki` is **Simple** English Wikipedia
(2 shards). **No official per-domain token table for v1.5 exists**, so natural weights are
measured from the data rather than looked up.

### Where the data and model come from

| what | link |
| ---- | ---- |
| Shard URL manifest (source of truth for every filename) | https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_5.txt |
| Shard host (the actual text) | https://olmo-data.org/dolma-v1_5r1/ |
| Dolma dataset card | https://huggingface.co/datasets/allenai/dolma |
| Base model + all 351 checkpoints | https://huggingface.co/allenai/OLMo-1B-hf |
| Checkpoint list (API) | https://huggingface.co/api/models/allenai/OLMo-1B-hf/refs |
| OLMo-1B original training config | https://github.com/allenai/OLMo/blob/v0.2.5/configs/official/OLMo-1B.yaml |

Access verified: public, ungated, ODC-BY, no token needed — HTTP 200 on all nine domains.
Example shard URL: `https://olmo-data.org/dolma-v1_5r1/wiki/en_simple_wiki_v0-0000.json.gz`
(note the filename is *not* `wiki-0000.json.gz` — constructing URLs 404s).

**Requests must send `Accept-Encoding: identity`.** The host layers HTTP gzip over the
already-gzipped shards for clients advertising gzip (Python `requests` does by default),
which drops Content-Length and corrupts the byte stream. `curl` works without this only
because it omits the header.

Papers behind the design: Skill-It https://arxiv.org/abs/2307.14430 ·
DoReMi https://arxiv.org/abs/2305.10429 · RegMix https://arxiv.org/abs/2407.01492 ·
Data Mixing Laws https://arxiv.org/abs/2403.16952 ·
DataDecide https://arxiv.org/abs/2504.11393 ·
T-LITE https://www.cs.utexas.edu/~lin/papers/isca24.pdf

## Arms — all five


| Arm | Weights from                                                  | Schedule                |
| --- | ------------------------------------------------------------- | ----------------------- |
| 1   | natural (Dolma's measured mix)                                | fixed                   |
| 2   | part-way runs → power-law extrapolation → LightGBM → minimize | fixed                   |
| 3   | same runs, data-mixing law instead of LightGBM                | fixed                   |
| 4   | Skill-It A_ij, full pairwise probing @~100M                   | adaptive, **5** updates |
| 5   | arm 4 + T-LITE k-means clustering, probing at cluster level   | adaptive, 5 updates     |


**Arms 2 and 3 share one proxy fleet.** The doc says arm 3 is "same as (2), but with the
data mixing law" — same part-way runs, different regressor. One fleet, two fits. This also
makes 2-vs-3 a clean comparison of regressor family rather than two unrelated pipelines.

**Arm 5's probing must be genuinely cheaper** or the arm proves nothing. Signatures come
from the arm-3 mixing-law fit — the doc offers either this or a row of A, and explicitly
rules out topic-embedding similarity — then k-means into K=4 clusters, then pairwise probing
*at cluster level*: **10 runs vs arm 4's 45**. Within a cluster, sampling follows natural
weighting. The 10-vs-45 figure is arm 5's *marginal* cost given that arm 3 is already being
run; standalone it also owes the 96-run fleet that produced the signatures, which makes it
more expensive than arm 4. Both are preregistered — see item 14.

**Not implemented, and declared:** the doc offers an optional extra for arm 5 — "for each
skill, only probe the top-N candidate prerequisite clusters by the derivative estimate."
Arm 5 as built probes all 10 cluster pairs. This overlaps with the registered derivative
follow-up below and is left to it (item 17).

**Probing cost** grows as k²/2. At k=9: 9 singles + C(9,2)=36 pairs = **45 runs**. The
training set {i,j} is unordered, so one run yields both A_ij and A_ji by evaluating on
both domains.

**Declared limitation** (in the doc's design, not a departure): arms 2/3 are fixed and 4/5
adaptive, but they also differ in *method*, so "adaptive wins" is partly confounded with
"Skill-It beats regression." Two of each makes it 2-vs-2 rather than 1-vs-1. Fully
isolating it would need a sixth arm — Skill-It weights held fixed. Reported, not fixed.

**Registered conditional follow-up** (the doc's): if the null is rejected, estimate A_ij
from the derivative of the data-mixing function w.r.t. one domain weight, and probe only
the highest estimated A_ij values. Success = less fitting compute than full pairwise, and
not significantly worse compute-to-loss.

## Scale


|                     | value                                                     | rationale                                                        |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------------------- |
| Main model          | OLMo-1B from `step1000-tokens4B`                          | doc                                                              |
| Proxy models        | 50M / 75M / 100M                                          | doc ("various 50-100M"; ~100M for arm 4)                         |
| Tokens per main run | **~2B**                                                   | Skill-It, the only continual-PT peer, used 1B and 3B             |
| Seeds per arm       | **3**                                                     | floor for the run-level variance the doc's statistical bar needs |
| Main runs           | 5 arms × 3 seeds = **15**                                 |                                                                  |
| Proxy fleet         | 32 Dirichlet mixtures × 3 sizes = 96 part-way runs [mine] | feeds arms 2 and 3                                               |
| Probe run length    | 200M tokens [mine]                                        |                                                                  |




## Dataset

**Cap 6B tokens per domain**, sized to a 100 GB disk budget [mine].


| domain       | pool    | note                                            |
| ------------ | ------- | ----------------------------------------------- |
| wiki         | 3.6B    | **all that exists** — measured, 2 shards, 6.0 GB |
| books        | 4.3B    | **all that exists** — measured, 3 shards, 7.1 GB |
| pes2o        | 6.0B    | 52B available (26 shards, 87 GB)                |
| reddit       | 6.0B    |                                                 |
| c4           | 6.0B    |                                                 |
| stack        | 6.0B    |                                                 |
| cc_en_head   | 6.0B    |                                                 |
| cc_en_middle | 6.0B    |                                                 |
| cc_en_tail   | 6.0B    |                                                 |
| **total**    | **50B** | ~83 GB download, ~93 GB disk (uint16)           |

wiki/books figures are from HEADing **every** shard. An earlier draft said 4.7B and 5.6B —
that sampled only the first shard of each domain and multiplied, which overstated both by
about 23%. Compressed bytes are now measured; token counts still use a ~600M-tokens/GB
prior until actual tokenization reports the true figure.


**No repetition.** Each run consumes ~2B tokens against a 50B pool = **0.040 epochs**, so
every token is seen at most once. Epochs over corpus for reference: DoReMi 0.350, RegMix
0.083, Data Mixing Laws 0.083, this plan 0.040 — the lowest of the group.

**Seeds read disjoint data** — `DomainPools` divides each pool into `n_seeds` regions and
starts each replicate in its own. Three seeds × 2B tokens with a realistic peak of about 40%
weight on one domain needs roughly 2.4B from that domain, giving 2.5× headroom on the seven
6B domains but only **1.5× on wiki (3.6B)** and 1.8× on books. Under a harder 80% peak,
wiki goes underwater and its seeds overlap. Mitigation is shorter runs (1.5B), not more disk
— wiki cannot grow. The trainer logs per-domain consumption and warns on pool exhaustion,
so this surfaces in run 1 rather than in the results.

**Known corpus limit, not a budget limit:** wiki (3.6B) and books (4.3B) are capped because
that is *all of them that exists* in Dolma v1.5 — the plan already takes 100% of both. No
amount of disk changes this. v1.5 `wiki` is Simple English Wikipedia, which is why it is so
small.

## Compute and storage required

Figures below are A100-equivalent GPU-hours at ~1.2e14 effective bf16 FLOP/s, counting
6ND per token. Both assumptions are optimistic; see the two corrections after the table.


| item                                             | GPU-h    |
| ------------------------------------------------ | -------- |
| Arm 4 A_ij, 45 probe runs @100M                  | 12       |
| Arm 5 cluster A_ij, 10 probe runs @100M          | 3        |
| Arms 2/3 shared fleet, 96 part-way runs @50–100M | 20       |
| **Main: 15 runs @1B, 2B tokens each**            | **490**  |
| **total**                                        | **~525** |


Two corrections that matter for planning:

**Gradient checkpointing costs ~33% and is not in the table.** `train_mixture.py` enables
it unconditionally, which recomputes the forward pass, so the main runs do ~8ND rather
than 6ND — 630 A100-hours, not 490. It is on because activation memory at batch 8 × 2048
was never measured on real hardware. If Phase 1 shows the model fits without it, turning
it off is the one cost saving that changes nothing scientific.

**"A100-equivalent" is doing real work in that sentence.** The experiment is planned for 4
H100s on an MIT ORCD group partition, where the 15 runs come to ~400 GPU-h and about 4.3
days of wall clock. The same runs on the L40S that ORCD hands out by default would be
~1240 GPU-h. See [RUNBOOK.md](RUNBOOK.md) for the current numbers, or run
`timing_estimate.py`.

Runs are **independent and embarrassingly parallel** — one process per GPU, no
multi-GPU training, no interconnect requirement — so the hardware decision affects
duration, not total cost.

Cutting compute, if needed: 1B tokens/run instead of 2B halves the main-run cost and
stays inside Skill-It's demonstrated 1B–3B range, at the price of a shorter run in which
mixture effects are smaller and noisier. Below that, seeds or arms have to go.

**Storage: ~93 GB** for the tokenized pool, plus space for checkpoints. Must sit on a
real filesystem — `DomainPools` memory-maps the token files, so object storage cannot
be read directly at training time. Reads are largely sequential (cursors advance
monotonically), so fast local disk helps but is not critical.

**Checkpoint/resume is required** regardless of hardware: main runs are long, and any
interruptible or preemptible compute would otherwise lose a whole run.

## Hyperparameters

Stated in one place because review items 2–4 cannot be judged without the full set. Where a
value differs from OLMo-1B's own config, both are shown.


|                   | this plan                    | OLMo-1B original     | note                                    |
| ----------------- | ---------------------------- | -------------------- | --------------------------------------- |
| Sequence length   | 2048                         | 2048                 | matches                                 |
| Batch (sequences) | 64 (8 × accum 8)             | 2048                 | **32× smaller** — item 2                |
| Tokens/step       | 131k                         | 4.19M                | consequence of the above                |
| Peak LR           | 3e-4 constant                | 4e-4, decays to 4e-5 | **item 4**                              |
| Warmup steps      | 200                          | 2000                 | branch point is mid-warmup — **item 3** |
| Optimizer         | AdamW, wd 0.1, β (0.9, 0.95) | AdamW, wd 0.1        | matches                                 |
| Grad clip         | 1.0                          | 1.0                  | matches                                 |
| Precision         | bf16                         | bf16                 | matches                                 |
| Mirror-descent η  | 0.5                          | n/a                  | arbitrary — **item 6**                  |
| Reweight rounds   | 5                            | n/a                  | doc-specified                           |
| Val held-out      | 0.5% tail per domain         | n/a                  | reserved before sampling                |




## Components

Built and tested:

- `prep_dolma_domains.py` — per-domain uint16 pools. Parallel across shards; streams and
tokenizes on the fly; concatenates via memmap in 64M-token chunks so a 6B-token domain is
never resident in RAM; resumable. Measures natural weights as tokens-per-byte x HEAD-sampled
total bytes.
- `train_mixture.py` — one trainer for all five arms. Val loss on reserved held-out slices is
the DV; 5 reweighting rounds; cluster mode for arm 5; disjoint per-seed read offsets;
checkpoint/resume; pool-exhaustion warnings. Skill-It mirror descent unit-tested. Also
carries the machinery that lets a run survive a cluster with short job windows:
`--max-seconds` stops cleanly on a checkpoint, `--ckpt-every-seconds` bounds how much a
kill can cost regardless of throughput, and SIGTERM/SIGUSR1 are caught so a preempted job
checkpoints at a step boundary rather than mid-optimizer-update. `--device cpu` runs the
control flow without a GPU.
- `fit_aij.py` — arm 4 full pairwise (45 runs at k=9), arm 5 cluster-level (10 runs at K=4).
Fresh identically-seeded proxy per probe. Records fitting compute separately. Shardable
across GPUs via `--shard/--num-shards`, with `--assemble-only` to merge.
- `cluster_tlite.py` — k-means (k-means++ seeding, 25 restarts) over behavioural signatures
from the arm-3 mixing-law fit, not topic embeddings. Verified to recover planted structure.
- `fit_proxy_fleet.py` — 32 Dirichlet mixtures x 3 sizes, loss curves recorded at 5 points
per run for power-law extrapolation. Mixtures drawn around the natural mix. Shardable on the
same interface as `fit_aij.py`.
- `extrapolate.py` — shared power-law fits (over tokens, then over model size), with guards:
falls back to last-observed when r2 < 0.5 or implied gain > 40%, and restricts weight search
to the per-domain range the fleet actually visited.
- `fit_regmix.py` — arm 2. LightGBM, falling back to sklearn GBM then ridge, reporting which
ran. Verified to recover a planted optimum on a synthetic fleet.
- `fit_mixing_law.py` — arm 3. Parametric law L_i = c_i + k_i exp(sum_j t_ij r_j), fitted by
scanning c and solving the linearised form. Also emits the `t` matrix arm 5 clusters on.
- `analyze.py` — sustained-crossing tokens-to-target across a swept grid, bootstrap CIs,
non-inferiority test against a preregistered margin, fitting compute reported separately.
- `PREREG.md` — to file before any main-run spend.

Operational layer (written, cluster-untested):

- `README.md` — arms, pipeline order, producer→consumer table, do-not-fix list.
- `RUNBOOK.md` — phased operating instructions for whoever holds the GPUs.
- `requirements.txt` — pinned floors; torch must be installed first against the cluster's
CUDA. LightGBM is listed but optional, since `fit_regmix.py` falls back and records which
regressor ran.
- `eval_benchmarks.py` — optional OLMES-style pass over the finished runs: nine multiple-
choice tasks ranked by length-normalised log likelihood, four metrics per task (accuracy,
per-char accuracy, and two continuous correct-probability measures) reported against each
task's chance rate. Inference only, feeds nothing in `analyze.py`. MMLU excluded and
declared, because OLMES scores it few-shot and these models have no in-context-learning
ability to measure. Scoring verified offline against a controlled model: continuation-only
slicing, per-token accumulation, padding invariance, and every task's gold index.
- `orcd_phase0_prep.sh` — CPU: estimate table, then build the pools into scratch.
- `orcd_phase1_smoke.sh` — the short GPU run, including a resume round-trip check, and it
prints measured tokens/s converted into a projected cost for the whole main phase. That
measurement is what replaces the estimates in this section.
- `orcd_phase2_fit.sh` — submitter that wires the real dependency graph: fleet and
arm-4 probe as parallel Slurm arrays, each followed by a merge job, then the CPU fitters and
clusterer, then the arm-5 array. `NSHARDS` sets the GPU width. Refuses to run until item 13
is acknowledged.
- `orcd_phase3_main.sh` — self-chaining array over the 15 main runs. No main run fits in
any ORCD public-partition window (6 h on `mit_normal_gpu`, 48 h on `mit_preemptable`
against ~27–83 h per run), so each task trains to `--max-seconds`, checkpoints, and
resubmits itself; `--requeue` plus `--signal=USR1@180` cover preemption. `--status` reports
progress across all 15.

Every `--flag` in these was machine-checked against the scripts' argparse definitions
(one real error caught: `--nproc` does not exist, the flag is `--procs`), and all four pass
`bash -n`. `.gitattributes` pins `*.sh` to LF so a Windows checkout cannot ship a CRLF
shebang to the cluster. **Slurm behaviour itself is unverified** — the directives, partition
names and limits come from the ORCD docs, not from a submitted job.

**All testing so far has been synthetic data on CPU.** No script has run against a real GPU,
the real model, or real tokens. What is verified is that the maths recovers known planted
answers and that each script's output format is what the next one reads. What is *not*
verified is that OLMo-1B loads, fits in memory, trains at the assumed throughput, or that
checkpoint/resume round-trips real weights.

Fixed since first draft:

- `prep_dolma_domains.py` now **tokenizes in parallel** — one worker process per shard,
streaming and tokenizing independently into part files, then concatenated in shard order
and truncated to target. Single-core prep of a 50B pool was hours of wall clock.
- `DomainPools` now takes `--seed-index`/`--n-seeds` and gives each replicate a **disjoint
starting offset**. Previously every seed started at position 0, so all three read the *same*
tokens and differed only in interleaving — not independent replicates. Verified: offsets
distinct, state round-trips including RNG, wraps counted at pool end.
- `train_mixture.py` **checkpoint/resume** — saves model, optimizer, scheduler, step, token
count, per-domain cursors, wrap counts, current weights and the next-eval/next-round
boundaries; reloads the model from the checkpoint rather than the base revision. Without
cursors and weights a restart would replay the same tokens and reset the reweighting
schedule, silently changing the experiment instead of resuming it. `test_resume.py`
exercises this against the real `main()` on a stubbed model: it stops a run early, resumes
it, and asserts that the token count moves forward, the step count moves forward, and the
pool cursors advance by exactly one batch per accumulation microstep with nothing reread.
It also checks that re-running a finished run is a no-op, since Phase 3 resubmits tasks.
- *Not* needed, after checking: a per-domain cap. A single flat target already self-caps —
wiki and books run out of shards and stop, keeping everything that exists, while the other
seven stop at target.

Bugs found by running the code, not by reading it:

- **HTTP gzip double-encoding.** olmo-data.org layers HTTP gzip on the already-gzipped
shards when a client advertises `Accept-Encoding: gzip` (requests' default). That dropped
Content-Length so all HEAD sizing silently returned nothing, AND corrupted the byte stream
(`BadGzipFile`). curl worked only because it omits that header. Fixed with
`Accept-Encoding: identity`. Would have broken the entire data pipeline.
- **wiki/books overstated by ~23%.** Sampling only the first shard and multiplying gave
4.7B/5.6B; HEADing every shard gives 3.6B/4.3B.
- **Manifest recorded `<domain>.npy.npy`**, which `DomainPools` could not have loaded.
- **Concatenation loaded a whole domain into RAM** (12 GB) before writing.
- **Extrapolation invented improvements from noise.** A flat noisy curve fitted with r2=0.09
still returned a 36% loss improvement, marked trustworthy. Now guarded.
- **Weight search escaped its own fit.** The ridge fallback predicted 0.033 loss — impossible
for a language model — at a simplex corner; the mixing law predicted below every value ever
observed. Both now constrained to the fleet's visited range.
- **Target grid swept a dead zone.** Under sustained crossing the slowest arm could not reach
any target, so the comparison table came out empty. Grid now spans only the band every run
can sustain-cross: 12 comparisons where there had been 1.

## Choices the doc does not specify

1. Dolma v1.5 as the corpus — follows from "OLMo-1B" but is never stated
2. k=9 domains (Dolma's own subsets, CC split into three quality tiers)
3. 6B tokens/domain cap → 50B pool (93 GB disk); ~2B tokens per main run
4. 3 seeds per arm
5. 32 Dirichlet mixtures × 3 proxy sizes; 200M tokens per probe run
6. K=4 clusters for arm 5
7. The non-inferiority margin for "not statistically significantly worse" — the doc states
  the bar but not the threshold, so it must be fixed in the prereg before running



## Status and what remains

Nothing has been downloaded and no GPU has been used. Total spend so far is zero.

### Blocking, before any spend

- **Item 13 first.** The A_ij probe holds total tokens constant rather than j's tokens,
which biases every entry of A negative; the Skill-It clip then zeroes them and arms 4 and 5
silently reproduce their starting weights. That is a null result manufactured by the
estimator, and it looks exactly like a real one. Nothing else on this list can invalidate
the experiment so completely or so quietly. The fix costs about 10 extra GPU-h.
- **The other 17 review questions.** Items 2–4 (batch size, mid-warmup branch point,
constant LR) and item 11 (bf16 optimiser state) are design decisions, not bugs; no amount
of code changes them. Getting them wrong wastes the entire main phase.
- **Items 5 and 12 are the two that DataDecide turned from open questions into arguable
errors**, and both are about the proxies rather than the main runs. The proxies are trained
at a token-to-parameter ratio of 2–3 where no one has shown mixture signal exists, and all
three sizes share one learning rate, which corrupts the size trend arm 3 extrapolates along.
Shrinking the proxies to ~10–20M parameters fixes the first for free; adopting per-size LR,
or dropping to a single size, fixes the second. Both change the fitting stage only, so they
are cheap to decide now and expensive to discover later.
- **Items 14–16 are decisions about what gets measured**, and all three have to be settled
before the prereg is filed rather than after: how arm 5's fitting cost is charged (which
decides whether it passes its own criterion), whether 3 seeds can detect the doc's stated
30–35% effect, and whether a benchmark pass is added to address the gap the doc itself
identifies.
- **The nine `[SET BEFORE LAUNCH]` values in `PREREG.md`** — token budget, seeds,
fleet size, probe length, K, dataset scope, and the non-inferiority margin. Values are
proposed there; a prereg with blanks is not a prereg. DataDecide now supplies real
precedent for one of them: it ran 3 seeds at its own 1B target for exactly this reason.

### Then, in order

1. `prep_dolma_domains.py --estimate-only` — natural-weight table, no download, no GPU.
2. **One short run on a real GPU.** A few minutes and a couple of dollars, and the highest
value step on this list: it is the first contact with the real model. Out-of-memory,
throughput below the assumed 1.2e14 FLOP/s, and checkpoint/resume against real weights are
all untested, and six of the seven bugs found so far only appeared when code actually ran.
3. Download and tokenize the pool — ~83 GB, several hours of wall clock.
4. Verification runs: short runs in each mode, confirming adaptive actually shifts
proportions across the 5 rounds (`weight_log.jsonl`), per-domain val loss is logged, and
pool-exhaustion warnings fire when forced.
5. Fitting runs — 45 probes for arm 4, 10 for arm 5, 96 for the arms 2/3 fleet (~35 GPU-h).
Sanity-check that A_ij is not degenerate (all-zero or all-equal) before trusting arms 4/5.
6. Run the three fitters (`fit_regmix`, `fit_mixing_law`, `cluster_tlite`) — CPU, minutes.
These produce the actual weight vectors for arms 2, 3 and 5.
7. File the prereg.
8. The 15 main runs (~400 GPU-h on H100s, ~4.3 days across four), then `analyze.py`.

