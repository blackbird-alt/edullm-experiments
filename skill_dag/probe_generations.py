"""
Diagnostic probe: print RAW generations from a checkpoint under three framings.
Resolves: loss=0.40 (learned) vs eval~0 (can't answer). RUN ON GPU, ~3 min.

  python probe_generations.py --model runs/smoke_random201/ckpt_tokens200015872_final

Framings per prompt:
  A) bare prompt                      "7 + 6 ="
  B) EOS-prefixed                     "<eos>7 + 6 ="   (packed-stream boundary)
  C) two training records + prompt    "a + b = c<eos>d + e = f<eos>7 + 6 ="
If A fails but C works, eval needs in-context framing (query mismatch).
If all fail, training itself produced a stream-reciter (overfit) or worse.
Also prints teacher-forced argmax check: given full训 "prompt+answer", are the
answer tokens the model's argmax? (True = knowledge is in there.)
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "skill_dag_dataset")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
model.eval()
eos = tok.eos_token or ""

# grab 2 training records per skill for context + 1 heldout prompt per skill
train_by, held_by = {}, {}
with open(os.path.join(DATA, "train.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        train_by.setdefault(r["skill"], []).append(r)
with open(os.path.join(DATA, "heldout.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        held_by.setdefault(r["skill"], r)  # first one

def gen(text, n=16):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return repr(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False))

for skill in ["A", "M", "ADD", "MUL", "DIV", "WORD"]:
    h = held_by[skill]
    t1, t2 = train_by[skill][0], train_by[skill][1]
    prompt = h["prompt"].rstrip()
    print(f"\n=== {skill}  expect: {h['answer']!r}")
    print(f"  A bare      : {gen(prompt)}")
    print(f"  B eos-prefix: {gen(eos + prompt)}")
    print(f"  C in-context: {gen(t1['text'] + eos + t2['text'] + eos + prompt)}")
    # teacher-forced: is each answer token the argmax after 'prompt'?
    full = tok(h["text"], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    plen = tok(h["prompt"].rstrip(), add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(full).logits
    pred = logits[0, :-1].argmax(-1)
    tgt = full[0, 1:]
    ans_slice = slice(plen - 1, full.shape[1] - 1)
    ok = (pred[ans_slice] == tgt[ans_slice]).float().mean().item()
    print(f"  teacher-forced answer-token argmax match: {ok:.2f}")
