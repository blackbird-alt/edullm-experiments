"""What this experiment costs on MIT ORCD hardware, and how long it takes to run.

Produces the tables in RUNBOOK.md. Everything here is an ESTIMATE from peak FLOPs and an
assumed utilisation -- the numbers move by a factor of two or more depending on that
assumption, which is why orcd_phase1_smoke.sh measures throughput for real before anyone
commits to the main phase. Re-run this after Phase 1 with MFU set to what was measured.

  python timing_estimate.py
"""
N = 1.18e9          # OLMo-1B-hf parameters
D_MAIN = 2.0e9      # tokens per main run (BUDGET default)
RUNS = 15
PROBES = 141        # 45 arm-4 + 10 arm-5 + 96 fleet (proxies, 50-100M)
N_PROXY = 0.075e9   # average proxy size across the 50/75/100M fleet
D_PROBE = 200e6

# dense BF16 tensor-core peak, and a realistic sustained fraction of it for a ~1B
# dense transformer. L40S is derated: 864 GB/s of memory bandwidth against the
# A100's 2039 makes it bandwidth-bound well before it is compute-bound.
GPUS = {
    "L40S  (44GB, default, 252 in mit_normal_gpu)": (181e12, 0.35),
    "A100  (preemptable only)":                     (312e12, 0.40),
    "H100  (80GB, 4 in mit_normal_gpu)":            (495e12, 0.40),
    "H200  (140GB, 88 in mit_normal_gpu)":          (495e12, 0.40),
}

print("Per main run = 2B tokens on a 1.18B model.")
print("6ND is fwd+bwd. Gradient checkpointing (currently ON) recomputes the forward,")
print("so the hardware actually does ~8ND -- a ~33% surcharge the old 490 GPU-h")
print("estimate did not include.\n")

f6 = 6 * N * D_MAIN
f8 = 8 * N * D_MAIN
print(f"  6ND = {f6:.3e} FLOPs      8ND = {f8:.3e} FLOPs\n")

hdr = f"{'GPU':46s} {'h/run':>7s} {'h/run':>7s} {'main':>7s} {'fit':>6s} {'TOTAL':>7s}"
print(hdr)
print(f"{'':46s} {'ckpt on':>7s} {'ckpt off':>7s} {'GPU-h':>7s} {'GPU-h':>6s} {'GPU-h':>7s}")
print("-" * len(hdr))
totals = {}
for name, (peak, mfu) in GPUS.items():
    eff = peak * mfu
    h_on = f8 / eff / 3600
    h_off = f6 / eff / 3600
    main = h_on * RUNS
    fit = PROBES * (6 * N_PROXY * D_PROBE) / eff / 3600     # proxies: no grad ckpt
    totals[name] = (h_on, main + fit)
    print(f"{name:46s} {h_on:7.1f} {h_off:7.1f} {main:7.0f} {fit:6.0f} {main+fit:7.0f}")

print("\n\nWALL CLOCK, given ORCD partition limits")
print("  mit_normal_gpu : 6 h max, 2 GPUs concurrent")
print("  mit_preemptable: 48 h max, 4 GPUs concurrent, jobs can be killed")
print("  pi_<group>     : a PI/group partition, typically 7-14 days and no preemption\n")
for part, cap, ngpu in [("mit_normal_gpu", 6, 2), ("mit_preemptable", 48, 4),
                        ("pi_<group>, 4 owned GPUs", 7 * 24, 4)]:
    print(f"  --- {part} ---")
    for name, (h_run, total) in totals.items():
        if part == "mit_normal_gpu" and "A100" in name:
            continue                        # A100 is preemptable-only
        chunks = -(-h_run // cap)
        days = total / ngpu / 24
        print(f"    {name:46s} {chunks:3.0f} chunks/run  {days:6.1f} days of compute")
    print()

print("Days above are pure compute and exclude queue wait, which is charged once per")
print("chunk. On mit_normal_gpu at 14 chunks/run that is 210 separate queue waits, which")
print("is why the public-partition L40S row is worse in practice than it looks here. On an")
print("owned partition there is no queue and one chunk per run, so the number is real.")
