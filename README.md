# eduLLM P1 experiments

Corpus-lever experiments for the eduLLM program (P1, Corpus Foundry track).

- `skill_dag/` — **the Skill-DAG experiment.** Ordering data by dependency
  structure rather than naive easy-to-hard. Does pre-training with adaptive domain
  weights need less compute to reach a given validation loss than fixed weights?
  Five arms (natural, RegMix, Data Mixing Laws, Skill-It A_ij, A_ij + T-LITE
  clustering) over Dolma v1.5. Base: OLMo-1B-hf @ `step1000-tokens4B`. 15 runs.
  **Not yet started** — no data downloaded, no GPU used; 18 review questions open.
  Design and open questions in `skill_dag/PLAN.md`; what has to be decided before
  launch is in `skill_dag/DECISIONS.md`; to actually run it, `skill_dag/RUNBOOK.md`.
  Launchers target MIT ORCD (Engaging) Slurm partitions.
- `base_370m/` — from-scratch training pipeline for a shared ~370M base
  (fallback; current plan uses allenai/OLMo-Ladder-760M-0.5xC, revision pinned
  in the runbook).
- `archive/skill_dag_arithmetic_ordering/` — the first Skill-DAG attempt, testing
  prerequisite *ordering* on synthetic arithmetic. Did not work: the model learned
  the number format rather than the arithmetic, then memorised the stream without
  extractable knowledge, and the hard skills floored below the mastery threshold.
  All three are consequences of the synthetic-arithmetic setup. Kept for reference;
  do not build on it. Post-mortem in `skill_dag/PLAN.md`.

## Data and weights are not in this repo

Nothing large is committed. Everything the code needs is public and fetched by scripts
that ship here, so a clone plus `skill_dag/requirements.txt` is enough to start:

| what | size | where it comes from |
|---|---|---|
| Dolma v1.5 shards | ~87 GB | `olmo-data.org`, via the 3,221 URLs in `skill_dag/dolma_v1_5_manifest.txt`; downloaded and tokenized by `prep_dolma_domains.py` |
| OLMo-1B-hf @ `step1000-tokens4B` | ~2 GB | Hugging Face, public and ungated, pulled automatically on first use |
| benchmark task data | small | Hugging Face, via `eval_benchmarks.py --download-only` |

The manifest itself came from `huggingface.co/datasets/allenai/dolma` (`urls/v1_5.txt`)
and is committed so the domain mix is reproducible rather than whatever the URL list
happens to say later. Run the downloads on a login node if your compute nodes are
offline — `skill_dag/RUNBOOK.md` covers that case for all three.
