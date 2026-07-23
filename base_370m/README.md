# Shared 370M base ("1xC")

The one starting model four P1 experiments branch from (scaffolding, QA-aug,
skill-DAG, difficulty-selection). Train once, freeze, share the path.
OLMo-2 architecture, 371.3M params (verified), OLMo-2 tokenizer, 1024 ctx,
1xC = ~7.4B tokens (~20 tokens/param) of the OLMo-2 pretraining mix.

## BEFORE launching
Post in the team channel: "training the shared 370M base now, will share the
path" — so it only gets trained once. (This base idea came from the scaffolding
thread; it serves the other levers too.)

## Run (on the cluster)

1. Data prep (CPU, needs internet, ~30 GB disk):
```bash
python prepare_data.py --n-tokens 7400000000 --out data/
```

2. Train (single node; 8 GPUs shown — single GPU also works, just slower):
```bash
torchrun --nproc_per_node 8 train_base.py --data data/ --out base_370m_1xC
```
Rough wall-clock: ~5-7 h on 8xA100, ~2 h on 8xH100. Progress + ETA print every
500 steps; loss log in `base_370m_1xC/train_log.jsonl`.

3. Freeze + share: the finished model is `base_370m_1xC/final`. Post that path.
   Early checkpoints (100M, 500M, then every 1B tokens) are saved automatically —
   the data-mixture experiment needs a barely-trained one, so keep them.

## Verified locally (CPU smoke test: `python smoke_test.py`)
- config instantiates at 371.3M params
- 3 real optimizer steps through the exact batch/loss/step code path
- checkpoint save/reload round-trip
- config eos/pad ids checked against the real OLMo-2 tokenizer (Olmo2Config
  defaults are OLMo-1 ids — explicitly overridden)

## Not yet verified (first cluster run will tell)
- olmo-mix-1124 streaming availability/auth on your cluster's network
- multi-GPU DDP path (written standard, but untested here)
- actual throughput (the ETA above is an estimate)

## Downstream contract
Every experiment that uses this base must reference the frozen `final` path and
never fine-tune it in place. Cosine LR decay is used HERE (standard pretraining);
the ordering experiments deliberately use constant LR on top of this base — that
difference is intentional, not an inconsistency.
