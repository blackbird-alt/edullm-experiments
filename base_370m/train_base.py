"""
From-scratch pretraining of the shared ~370M base (OLMo-2 architecture).
This is the ONE artifact four P1 experiments (scaffolding, QA-aug, skill-DAG,
difficulty-selection) branch off. Train once, freeze, share the path.

- Reads the frozen token file from prepare_data.py (no streaming at train time).
- Cosine LR decay IS correct here (standard pretraining); the constant-LR rule
  applies only to the ordering experiments built on top of this base.
- Saves early checkpoints too (100M, 500M, then every 1B tokens) — the data-
  mixture experiment wants a barely-trained checkpoint, so we get it for free.

Launch (single node, 8 GPUs):
  torchrun --nproc_per_node 8 train_base.py --data data/ --out base_370m_1xC
Single GPU works too (just slower):  python train_base.py --data data/ --out ...
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

TOKENIZER = "allenai/OLMo-2-0425-1B"

# ~370M-parameter OLMo-2 config (tied embeddings). Verified by smoke_test.py.
CFG = dict(
    vocab_size=100352, hidden_size=1024, num_hidden_layers=16,
    num_attention_heads=16, num_key_value_heads=16, intermediate_size=4096,
    max_position_embeddings=1024, tie_word_embeddings=True,
    # OLMo-2 tokenizer ids (Olmo2Config defaults are OLMo-1's — wrong for our tokenizer)
    eos_token_id=100257, pad_token_id=100277,
)
SEQ_LEN = 1024
CKPT_MARKS_M = [100, 500] + [1000 * i for i in range(1, 8)]  # in millions of tokens


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing train.bin + manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32, help="per-GPU micro-batch")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    torch.manual_seed(args.seed + rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data, "manifest.json")) as f:
        manifest = json.load(f)
    n_tokens = manifest["n_tokens"]
    data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint32,
                     mode="r", shape=(n_tokens,))
    n_seqs = n_tokens // SEQ_LEN

    model = Olmo2ForCausalLM(Olmo2Config(**CFG)).to(device, dtype=torch.bfloat16)
    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"params: {n_params/1e6:.1f}M   data: {n_tokens/1e9:.2f}B tokens "
              f"({n_seqs:,} seqs of {SEQ_LEN})   world={world}")
    if world > 1:
        model = DDP(model)

    tokens_per_step = args.batch_size * SEQ_LEN * args.grad_accum * world
    total_steps = n_tokens // tokens_per_step
    warmup = max(1, int(total_steps * args.warmup_frac))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    def lr_at(step):  # warmup + cosine to 10% of peak
        if step < warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    os.makedirs(args.out, exist_ok=True)
    marks = [m * 1_000_000 for m in CKPT_MARKS_M]
    seen_tokens, step, t0 = 0, 0, time.time()
    log = os.path.join(args.out, "train_log.jsonl")

    def get_batch(step_idx, accum_idx):
        # each rank reads a disjoint, deterministic slice of the sequence stream
        base = (step_idx * args.grad_accum + accum_idx) * args.batch_size * world
        idx = [(base + rank * args.batch_size + j) % n_seqs for j in range(args.batch_size)]
        x = np.stack([data[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in idx]).astype(np.int64)
        t = torch.from_numpy(x).to(device)
        return {"input_ids": t, "labels": t.clone()}

    model.train()
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for a in range(args.grad_accum):
            loss = model(**get_batch(step, a)).loss / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        seen_tokens += tokens_per_step

        if is_main() and step % 50 == 0:
            tps = seen_tokens / (time.time() - t0)
            with open(log, "a") as f:
                f.write(json.dumps({"step": step, "tokens": seen_tokens,
                                    "loss": round(loss.item() * args.grad_accum, 4),
                                    "lr": lr_at(step), "tok_per_s": int(tps)}) + "\n")
            if step % 500 == 0:
                eta_h = (n_tokens - seen_tokens) / tps / 3600
                print(f"step {step}/{total_steps}  tokens {seen_tokens/1e9:.2f}B  "
                      f"loss {loss.item()*args.grad_accum:.3f}  ETA {eta_h:.1f}h", flush=True)
        if is_main() and marks and seen_tokens >= marks[0]:
            m = marks.pop(0)
            ck = os.path.join(args.out, f"ckpt_tokens{m}")
            (model.module if world > 1 else model).save_pretrained(ck)
            AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(ck)
            print(f"saved early checkpoint {ck}", flush=True)

    if is_main():
        final = os.path.join(args.out, "final")
        (model.module if world > 1 else model).save_pretrained(final)
        AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(final)
        with open(os.path.join(args.out, "base_config.json"), "w") as f:
            json.dump({"cfg": CFG, "seq_len": SEQ_LEN, "args": vars(args),
                       "data_manifest": manifest, "total_steps": total_steps,
                       "final_tokens": seen_tokens}, f, indent=2)
        print(f"DONE: frozen base at {final} ({seen_tokens/1e9:.2f}B tokens)")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
