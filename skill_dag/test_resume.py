"""Does --max-seconds stop on a good checkpoint, and does --resume continue from it?

This is the test that backs the Phase 3 design. No ORCD partition can hold a whole main
run -- 6 h on mit_normal_gpu, 48 h on mit_preemptable, against 27-83 h per run -- so
orcd_phase3_main.sh trains in chunks that stop, checkpoint, and resubmit. If stop/resume
is wrong, that either silently restarts runs from zero or replays the same tokens, and
either way the compute is wasted and the result is invalid.

Stubs the model, tokenizer and data pools so it runs on CPU in about a minute. save_ckpt,
load_ckpt and the whole main() control flow are the REAL ones -- that is the point.

  python test_resume.py        # prints ALL OK, or raises
"""
import json
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

V, H = 64, 16


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.head = nn.Linear(H, V)
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(self, input_ids=None, labels=None):
        x = self.head(self.emb(input_ids))
        loss = nn.functional.cross_entropy(x.view(-1, V), labels.view(-1))
        return types.SimpleNamespace(loss=loss, logits=x)

    def gradient_checkpointing_enable(self, **k):
        pass

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(d, "weights.pt"))
        json.dump({"model_type": "tiny"}, open(os.path.join(d, "config.json"), "w"))


def _from_pretrained(path, **kw):
    m = TinyModel()
    w = os.path.join(str(path), "weights.pt")
    if os.path.exists(w):
        m.load_state_dict(torch.load(w, map_location="cpu"))
    return m


tf = types.ModuleType("transformers")
tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=_from_pretrained)
tf.AutoTokenizer = types.SimpleNamespace(
    from_pretrained=lambda *a, **k: types.SimpleNamespace(
        save_pretrained=lambda d: os.makedirs(d, exist_ok=True)))
tf.get_constant_schedule_with_warmup = lambda opt, num_warmup_steps=0: \
    torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
sys.modules["transformers"] = tf

import train_mixture as T                                        # noqa: E402

DOMAINS = ["a", "b", "c"]
STEP_TOKENS = 2 * 2 * 256          # batch 2 x accum 2 x seq 256


class FakePools:
    domains = DOMAINS
    natural = {d: 1 / 3 for d in DOMAINS}

    def __init__(self, *a, **k):
        self.start = {d: 0 for d in DOMAINS}
        self.wraps = {d: 0 for d in DOMAINS}
        self.cursor = {d: 0 for d in DOMAINS}

    def batch(self, weights, bs, seq):
        for d in DOMAINS:
            self.cursor[d] += 1
        return torch.randint(0, V, (bs, seq)), {d: (bs * seq) // 3 for d in DOMAINS}

    def state(self):
        return {"cursor": dict(self.cursor), "wraps": dict(self.wraps),
                "start": dict(self.start)}

    def load_state(self, s):
        self.cursor = s["cursor"]; self.wraps = s["wraps"]; self.start = s["start"]


T.DomainPools = FakePools
T.val_loss = lambda *a, **k: dict({d: 1.0 for d in DOMAINS}, _mean=1.0)
T.load_weights = lambda spec, pools: {d: 1 / 3 for d in DOMAINS}


def run(out, budget, *extra, ckpt_s=0):
    sys.argv = ["train_mixture.py", "--arm", "test", "--out", out, "--device", "cpu",
                "--weights", "natural", "--reweight-mode", "fixed",
                "--token-budget", str(budget), "--batch-size", "2", "--grad-accum", "2",
                "--seq-len", "256", "--eval-tokens", str(budget * 10),
                "--ckpt-every-tokens", "0", "--ckpt-every-seconds", str(ckpt_s),
                "--warmup-steps", "1", *extra]
    T.main()


def state(out):
    p = os.path.join(out, "ckpt_resume", "progress.json")
    return json.load(open(p)) if os.path.exists(p) else None


def finished(out):
    return os.path.exists(os.path.join(out, "run_config.json"))


# A budget far larger than a 2-second chunk can consume, so "stopped early" is not a
# race against how loaded this machine happens to be.
BIG = STEP_TOKENS * 20000
tmp = tempfile.mkdtemp()

print("=== --max-seconds stops early and leaves a checkpoint ===")
a = os.path.join(tmp, "a")
run(a, BIG, "--max-seconds", "2")
st = state(a)
assert st is not None, "stopped without writing a checkpoint"
assert not finished(a), "wrote run_config.json despite stopping early"
assert 0 < st["trained"] < BIG
print(f"  stopped at {st['trained']:,}/{BIG:,} tokens, step {st['step']}")

print("\n=== --resume continues from there rather than restarting ===")
run(a, BIG, "--max-seconds", "2", "--resume")
st2 = state(a)
assert not finished(a), "test budget was too small to stay unfinished"
assert st2["trained"] > st["trained"], \
    f"resume restarted: {st2['trained']:,} is not past {st['trained']:,}"
assert st2["step"] > st["step"]
print(f"  reached {st2['trained']:,} tokens at step {st2['step']} "
      f"(was {st['trained']:,} at step {st['step']})")

print("\n=== data cursors advance across the resume, so tokens are not replayed ===")
before, after = st["pools"]["cursor"]["a"], st2["pools"]["cursor"]["a"]
assert after > before, "pool cursor rewound -- the resume would retrain the same tokens"
GRAD_ACCUM = 2                       # batch() is called once per accumulation microstep
assert after - before == (st2["step"] - st["step"]) * GRAD_ACCUM, \
    f"cursor moved {after - before} for {st2['step'] - st['step']} steps"
print(f"  cursor {before} -> {after}, exactly {GRAD_ACCUM} batches per step, none reread")

print("\n=== chunked training reaches the budget and writes final/ ===")
b = os.path.join(tmp, "b")
small = STEP_TOKENS * 40
chunks = 0
for _ in range(30):
    if finished(b):
        break
    run(b, small, "--max-seconds", "1", *(["--resume"] if chunks else []))
    chunks += 1
assert finished(b), f"never completed after {chunks} chunks"
cfg = json.load(open(os.path.join(b, "run_config.json")))
assert cfg["final_tokens"] >= small, f"completed short at {cfg['final_tokens']}"
assert os.path.exists(os.path.join(b, "final", "config.json")), "no final/ model"
print(f"  {cfg['final_tokens']:,}/{small:,} tokens over {chunks} chunk(s), "
      f"{cfg['steps']} steps, final/ written")

print("\n=== a completed run is a no-op, so a stray resubmission cannot corrupt it ===")
run(b, small, "--max-seconds", "1", "--resume")
cfg2 = json.load(open(os.path.join(b, "run_config.json")))
assert cfg2["final_tokens"] == cfg["final_tokens"], "re-running a finished run changed it"
print(f"  still {cfg2['final_tokens']:,} tokens")

print("\n=== time-based cadence checkpoints without a token cadence ===")
c = os.path.join(tmp, "c")
run(c, BIG, "--max-seconds", "4", ckpt_s=1)
assert state(c) is not None
print(f"  checkpoint at {state(c)['trained']:,} tokens with --ckpt-every-tokens 0")

print("\n=== SIGTERM is caught and checkpoints rather than dropping work ===")
if os.name == "nt":
    # Windows cannot deliver a Slurm-style SIGTERM to itself; the handler shares the
    # stop-and-checkpoint branch exercised above by --max-seconds.
    print("  skipped on Windows -- --max-seconds covers the same branch")
else:
    import signal
    import threading
    d = os.path.join(tmp, "d")
    threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    run(d, BIG)
    assert state(d) is not None and not finished(d)
    print(f"  SIGTERM at {state(d)['trained']:,} tokens, checkpoint written")

print("\nALL OK")
