# Skill-DAG

Does pre-training with **adaptive** domain weights reach a given validation loss with less
compute than **fixed** domain weights? Weights come from Skill-It's A_ij adjacency matrix —
how much training on domain *i* improves domain *j*.

Full design and the open questions: [PLAN.md](PLAN.md). Decision rule and what is frozen
before launch: [PREREG.md](PREREG.md). Operating instructions for whoever has the GPUs:
[RUNBOOK.md](RUNBOOK.md).

Supersedes an earlier attempt on synthetic arithmetic, archived at
`../archive/skill_dag_arithmetic_ordering/`. Why it failed and why this design avoids that
failure class: see "Previous attempt" in PLAN.md.

## Arms

| Arm | Weights from | Schedule |
|---|---|---|
| 1 | natural (Dolma's measured mix) | fixed |
| 2 | proxy fleet → power law → LightGBM → minimise | fixed |
| 3 | same fleet, data-mixing law instead of LightGBM | fixed |
| 4 | Skill-It A_ij, full pairwise probing (45 runs) | adaptive, 5 updates |
| 5 | arm 4 + T-LITE clustering, probing per cluster (10 runs) | adaptive, 5 updates |

3 seeds each = **15 main runs**. Arms 2 and 3 share one proxy fleet, so 2-vs-3 isolates
regressor family rather than comparing two unrelated pipelines.

## Pipeline

Run in this order. Producer → consumer is verified; each step reads the previous step's
output file.

| # | script | where | produces |
|---|---|---|---|
| 0 | `prep_dolma_domains.py` | CPU, ~87 GB download | `dolma_domains/` uint16 pools + natural weights |
| 1 | *smoke test* (RUNBOOK step 3) | GPU, minutes | first contact with the real model |
| 2a | `fit_proxy_fleet.py` | GPU, 96 runs | `fleet/fleet.jsonl` |
| 2b | `fit_aij.py` | GPU, 45 runs | `aij_arm4/aij.json` |
| 3a | `fit_regmix.py` | CPU, minutes | `weights_arm2.json` |
| 3b | `fit_mixing_law.py` | CPU, minutes | `weights_arm3.json`, `mixlaw_t.json` |
| 4 | `cluster_tlite.py` | CPU, seconds | `clusters.json` |
| 5 | `fit_aij.py --cluster-map` | GPU, 10 runs | `aij_arm5/aij.json` |
| 6 | `train_mixture.py` ×15 | GPU, ~490 GPU-h | `runs/*/val_log.jsonl` |
| 7 | `analyze.py` | CPU | `analysis.json` + verdict |

`extrapolate.py` is a shared library (power-law fits and support guards) imported by steps
3a and 3b, not run directly.

Step 5 depends on step 4, which depends on 3b, which depends on 2a. Step 2b is independent
and can run alongside 2a.

## Status

- [x] All 9 scripts written; interfaces verified producer → consumer
- [x] Logic verified against planted ground truth (power law recovers E=2.0 at r²=0.999999;
      both fitters recover a planted optimum; clustering recovers planted structure;
      analysis recovers a planted 3× advantage)
- [ ] **Nothing has touched a GPU or a real token.** All verification so far is synthetic
      data on CPU. The smoke test (step 1) is first contact.
- [ ] 18 review questions open — see PLAN.md. **Item 13 blocks step 2b**: the A_ij probe
      holds total tokens fixed rather than j's, which biases every entry negative and can
      freeze the adaptive arms at their starting weights, retaining the null by
      construction.
- [ ] 9 `[SET BEFORE LAUNCH]` values unset in PREREG.md
- [ ] Prereg not filed

## Do not "fix" these

Deliberate choices; changing them changes the experiment.

- **Constant LR, no decay.** Decay would down-weight late-arriving data, which in adaptive
  mode is exactly the data the reweighting selected — confounding the comparison.
- **Held-out validation slices are reserved before any sampling.** No arm can train on them.
- **`analyze.py` is committed before the runs.** Sustained-crossing detection, the target
  grid, the bootstrap and the non-inferiority test are all in it. No post-hoc threshold
  movement.
- **Fitting compute is reported separately from training compute**, never pooled — the
  doc's success bar for arm 5 and for the follow-up is about fitting cost.
- **Requests to olmo-data.org must send `Accept-Encoding: identity`.** The host double-gzips
  otherwise, which drops Content-Length and corrupts the stream. `curl` happens to work
  without it; Python `requests` does not.
