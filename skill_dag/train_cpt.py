"""
CPT training for the skill-DAG sequencing experiment. RUN ON GPU.

One invocation = one arm-run: continual pretraining from the shared frozen base,
consuming ONE schedule file (topo_XXX.idx or random_XXX.idx). The script is
byte-identical across arms; the schedule file is the only difference.

Design choices that protect the experiment (do not change casually):
- ORDER IS PRESERVED EXACTLY: records are packed into sequences in schedule
  order and batches are drawn sequentially. No dataloader shuffling anywhere.
- CONSTANT LR + WARMUP (no cosine): a decaying LR would down-weight whatever
  data appears late, confounding the order manipulation (the Luo et al. 2025
  "LR decay wastes your best data" problem). Constant LR treats every stream
  position equally, which is the fair test for ordering.
- MULTIPLE EPOCHS: if token_budget exceeds one pass, the same ordered stream
  repeats. The topological constraint holds within each pass.
- CHECKPOINTS EVERY --ckpt-tokens: mastery is measured as a curve over trained
  tokens (by eval_mastery.py on the saved checkpoints), not only at the end.

Usage (single GPU is enough at 370M-1B):
  python train_cpt.py --model <base> --schedule schedules/topo_101.idx \
      --token-budget 2000000000 --out runs/topo_101
"""
import argparse, json, math, os, time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "skill_dag_dataset")


class OrderedPackedDataset(Dataset):
    """Packs record texts into fixed-length blocks IN SCHEDULE ORDER."""

    def __init__(self, schedule_path, tokenizer, seq_len):
        with open(os.path.join(DATA, "train.jsonl")) as f:
            texts = [json.loads(l)["text"] for l in f]
        with open(schedule_path) as f:
            order = [int(x) for x in f.read().split()]
        eos = tokenizer.eos_token_id
        ids = []
        for idx in order:
            ids.extend(tokenizer(texts[idx], add_special_tokens=False)["input_ids"])
            ids.append(eos)  # record separator
        n_blocks = len(ids) // seq_len
        self.blocks = [ids[i * seq_len : (i + 1) * seq_len] for i in range(n_blocks)]
        self.tokens_per_epoch = n_blocks * seq_len

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        x = torch.tensor(self.blocks[i], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path or HF id of the frozen shared base")
    ap.add_argument("--schedule", required=True, help="schedules/<name>.idx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-budget", type=int, required=True)
    ap.add_argument("--ckpt-tokens", type=int, default=100_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--wandb", default=None,
                    help="optional W&B project (e.g. eduLLM/skill-dag); run name = schedule name")
    args = ap.parse_args()

    wb = None
    if args.wandb:
        import wandb as wb  # optional dep; only imported when requested
        entity_project = args.wandb.split("/")
        wb.init(entity=entity_project[0] if len(entity_project) > 1 else None,
                project=entity_project[-1],
                group="skill-dag-sequencing",
                name="skilldag-" + os.path.splitext(os.path.basename(args.schedule))[0],
                config=vars(args))

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()

    ds = OrderedPackedDataset(args.schedule, tok, args.seq_len)
    print(f"schedule={os.path.basename(args.schedule)}  tokens/epoch={ds.tokens_per_epoch:,}  "
          f"budget={args.token_budget:,}  (~{args.token_budget / ds.tokens_per_epoch:.1f} epochs)")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)  # shuffle=False is the experiment

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    trained_tokens, next_ckpt, step = 0, args.ckpt_tokens, 0
    log_path = os.path.join(args.out, "train_log.jsonl")
    t0 = time.time()

    model.train()
    done = False
    while not done:  # epochs over the same ordered stream
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                trained_tokens += tokens_per_step
                if step % 50 == 0:
                    rec = {"step": step, "tokens": trained_tokens,
                           "loss": round(loss.item() * args.grad_accum, 4),
                           "elapsed_s": round(time.time() - t0)}
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    if wb:
                        wb.log(rec, step=step)
                if trained_tokens >= next_ckpt:
                    ck = os.path.join(args.out, f"ckpt_tokens{trained_tokens}")
                    model.save_pretrained(ck); tok.save_pretrained(ck)
                    print(f"saved {ck}")
                    next_ckpt += args.ckpt_tokens
                if trained_tokens >= args.token_budget:
                    done = True
                    break
        if not done:
            print(f"epoch boundary at {trained_tokens:,} tokens; repeating ordered stream")

    final = os.path.join(args.out, f"ckpt_tokens{trained_tokens}_final")
    model.save_pretrained(final); tok.save_pretrained(final)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args) | {"tokens_per_epoch": ds.tokens_per_epoch,
                                "final_tokens": trained_tokens}, f, indent=2)
    print(f"done: {trained_tokens:,} tokens -> {final}")


if __name__ == "__main__":
    main()
