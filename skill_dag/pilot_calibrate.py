"""
Pilot: difficulty calibration for the skill-DAG experiment. RUN ON GPU.

Measures the BASE model's per-skill bare-answer accuracy on the held-out set,
before any CPT. Purpose:
  1. Catch floor/ceiling skills: if a skill is ~0% even after brief exposure
     (or ~100% cold), the main experiment can't measure an order effect on it
     -> tune that skill's digit ranges in build_skill_dag_dataset.py and refreeze.
  2. Baseline numbers for the prereg.

Usage:
  python pilot_calibrate.py --model <path-or-hf-id> [--n-per-skill 200] [--batch-size 32]

Output: pilot_results.json + a printed table with a FLOOR/CEILING/OK verdict per skill.
Notes: greedy decoding, exact match on the answer string, prompt = record's "prompt"
(e.g. "347 + 275 = "). Cold accuracy near 0 for a base model is EXPECTED on the
harder skills; the real gate is the main runs' early checkpoints. This pilot's
verdicts are a cheap first filter, not the final word.
"""
import argparse, json, os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "skill_dag_dataset")

FLOOR, CEILING = 0.02, 0.95  # verdict thresholds on cold accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-per-skill", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    by_skill = defaultdict(list)
    with open(os.path.join(DATA, "heldout.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if len(by_skill[r["skill"]]) < args.n_per_skill:
                by_skill[r["skill"]].append(r)

    results = {}
    for skill, recs in sorted(by_skill.items()):
        correct = 0
        for i in range(0, len(recs), args.batch_size):
            batch = recs[i : i + args.batch_size]
            prompts = [r["prompt"] for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, g in zip(batch, gen):
                # exact match on the leading answer token(s); tolerate trailing text
                pred = g.strip().split("\n")[0].strip()
                if pred.startswith(r["answer"]) and (
                    len(pred) == len(r["answer"]) or not pred[len(r["answer"])].isdigit()
                ):
                    correct += 1
        acc = correct / len(recs)
        verdict = "FLOOR" if acc < FLOOR else "CEILING" if acc > CEILING else "OK"
        results[skill] = {"n": len(recs), "accuracy": round(acc, 4), "verdict": verdict}
        print(f"{skill:6} n={len(recs):4d}  acc={acc:6.3f}  {verdict}")

    with open(os.path.join(HERE, "pilot_results.json"), "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)

    flagged = [s for s, r in results.items() if r["verdict"] != "OK"]
    if flagged:
        print(f"\nFlagged (cold): {flagged}")
        print("CEILING skills: make ranges harder in build_skill_dag_dataset.py, refreeze as v2.")
        print("FLOOR skills: expected cold for a base model; check whether they LIFT by the")
        print("first checkpoints of a short training smoke run before shrinking ranges.")


if __name__ == "__main__":
    main()
