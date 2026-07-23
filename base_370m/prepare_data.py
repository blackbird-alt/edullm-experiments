"""
Data prep for the shared ~370M base ("1xC" = ~20 tokens/param = 7.4B tokens).

Streams the OLMo 2 pretraining mix from HF, tokenizes with the OLMo-2 tokenizer,
and writes EXACTLY --n-tokens token ids to a uint32 memmap (train.bin). A frozen
token file (instead of streaming at train time) makes the base reproducible and
auditable: the manifest records source, seed, and a content hash.

Run on a machine with internet (login node is fine; this is CPU work):
  python prepare_data.py --n-tokens 7400000000 --out data/
Needs ~30 GB disk for 7.4B uint32 tokens. Use --dataset to swap sources if the
default is gated/unavailable on your cluster.
"""
import argparse, hashlib, json, os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = "allenai/OLMo-2-0425-1B"  # same tokenizer as the OLMo-2 family
DEFAULT_DATASET = "allenai/olmo-mix-1124"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id
    ds = load_dataset(args.dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    path = os.path.join(args.out, "train.bin")
    arr = np.memmap(path, dtype=np.uint32, mode="w+", shape=(args.n_tokens,))
    h = hashlib.sha256()
    filled, n_docs = 0, 0
    for doc in ds:
        text = doc.get("text") or ""
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids.append(eos)
        take = min(len(ids), args.n_tokens - filled)
        chunk = np.asarray(ids[:take], dtype=np.uint32)
        arr[filled : filled + take] = chunk
        h.update(chunk.tobytes())
        filled += take
        n_docs += 1
        if n_docs % 20_000 == 0:
            print(f"{filled:,}/{args.n_tokens:,} tokens ({n_docs:,} docs)", flush=True)
        if filled >= args.n_tokens:
            break
    arr.flush()
    assert filled == args.n_tokens, f"stream ended early at {filled:,} tokens"

    manifest = {
        "tokenizer": TOKENIZER, "dataset": args.dataset, "seed": args.seed,
        "n_tokens": args.n_tokens, "n_docs": n_docs, "dtype": "uint32",
        "sha256": h.hexdigest(),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
