"""
CPU smoke test for train_base.py — run anywhere, no GPU or data needed.
1. Instantiates the real 370M config and prints the parameter count.
2. Builds a tiny variant + fake token file and runs 3 real optimizer steps
   through the same batch/loss/step logic to catch API bugs before cluster time.
"""
import json, os, tempfile

import numpy as np
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from train_base import CFG, SEQ_LEN

# 1) real config param count
model = Olmo2ForCausalLM(Olmo2Config(**CFG))
n = sum(p.numel() for p in model.parameters())
print(f"full config params: {n/1e6:.1f}M  (target ~370M)")
del model

# 2) tiny end-to-end: 3 training steps on fake data
tiny = CFG | dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, intermediate_size=128, vocab_size=1000)
m = Olmo2ForCausalLM(Olmo2Config(**tiny))
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
rng = np.random.default_rng(0)
data = rng.integers(0, 1000, size=(16 * SEQ_LEN,), dtype=np.uint32)
losses = []
for step in range(3):
    idx = [(step * 2 + j) % 16 for j in range(2)]
    x = np.stack([data[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in idx]).astype(np.int64)
    t = torch.from_numpy(x)
    loss = m(input_ids=t, labels=t.clone()).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    losses.append(round(loss.item(), 3))
print(f"3 tiny train steps OK, losses: {losses}")

# 3) save/reload round-trip (what the checkpoint code does)
with tempfile.TemporaryDirectory() as d:
    m.save_pretrained(d)
    m2 = Olmo2ForCausalLM.from_pretrained(d)
    print("save/reload round-trip OK")
print("SMOKE TEST PASSED")
