"""
Build per-domain token pools from Dolma v1.5 for the Skill-DAG weighting experiment.

Why per-domain pools: the experiment reweights domains, so the training data MUST be
separable by domain. The convenient 10B-token `v1_6-sample` is a flat pre-mixed stream
with no per-subset structure, so it cannot be used here -- we slice the full v1_5
manifest by domain instead.

Design notes (do not change casually):
- URLS COME FROM THE MANIFEST, never constructed. Filenames differ per domain
  (wiki -> en_simple_wiki_v0-*.json.gz, pes2o -> pes2o_v2-*.json.gz), so building
  "<domain>-0000.json.gz" 404s for several domains.
- PARALLEL ACROSS SHARDS. Tokenizing ~50B tokens single-threaded is hours of wall
  clock. One worker per shard, each streaming + tokenizing independently into a part
  file; parts are then concatenated in shard order and truncated to target.
- STREAMING: shards are 1.3-4 GB gzipped (~1-3B tokens each). Decompressed and
  tokenized on the fly; the raw .json.gz is never written to disk.
- SMALL DOMAINS SELF-CAP. wiki has 2 shards (~3.6B tokens) and books 3 (~4.3B), which
  is ALL that exists in v1.5. With a target above that they simply take everything and
  stop -- no special-casing needed.
- uint16 token storage: OLMo's vocab is 50280 (embedding 50304), fits in uint16 and
  halves disk vs int32.
- NATURAL WEIGHTING: AI2 published no per-domain token table for v1.5 (the HF card's
  table is v1.6, and the OLMo paper's is older still and omits C4). Arm 1 needs real
  proportions, so we measure them: tokens-per-utf8-byte on the shards actually read,
  scaled by each domain's total compressed bytes (HEAD requests, no download).
- RESUMABLE: per-domain .npy + state file; re-running skips domains already at target.

Usage:
  python prep_dolma_domains.py --tokens-per-domain 6_000_000_000 --out dolma_domains
  python prep_dolma_domains.py --estimate-only      # natural-weight table, no download
"""
import argparse, gzip, io, json, multiprocessing as mp, os, time

import numpy as np
import requests

MANIFEST_URL = "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_5.txt"
BASE = "https://olmo-data.org/dolma-v1_5r1/"
TOKENIZER = "allenai/OLMo-1B-hf"      # ships the tokenizer used for Dolma v1.5
CC_TIERS = ("cc_en_head", "cc_en_middle", "cc_en_tail")
TOK_PER_GB_GZ = 600e6                  # conservative prior for shard-count estimation

# MUST send identity encoding. olmo-data.org applies HTTP gzip on top of the already
# gzipped .json.gz when a client advertises Accept-Encoding: gzip (which requests does by
# default). That (a) drops Content-Length so HEAD sizing silently returns nothing, and
# (b) corrupts the raw byte stream so gzip.GzipFile raises BadGzipFile. curl worked only
# because it does not send Accept-Encoding unless asked.
HEADERS = {"Accept-Encoding": "identity"}

_TOK = None                            # per-worker tokenizer


# ----------------------------------------------------------------- manifest


def fetch_manifest(cache="dolma_v1_5_manifest.txt"):
    if os.path.exists(cache):
        with open(cache) as f:
            return [l.strip() for l in f if l.strip()]
    r = requests.get(MANIFEST_URL, timeout=60)
    r.raise_for_status()
    urls = [l.strip() for l in r.text.splitlines() if l.strip()]
    with open(cache, "w") as f:
        f.write("\n".join(urls))
    return urls


def group_by_domain(urls, merge_cc=False):
    out = {}
    for u in urls:
        rest = u[len(BASE):] if u.startswith(BASE) else u.split("/", 3)[-1]
        dom = rest.split("/")[0]
        if merge_cc and dom in CC_TIERS:
            dom = "cc_en"
        out.setdefault(dom, []).append(u)
    return out


def head_bytes(url, retries=3):
    for a in range(retries):
        try:
            r = requests.head(url, allow_redirects=True, timeout=30, headers=HEADERS)
            if r.status_code == 200 and "content-length" in r.headers:
                return int(r.headers["content-length"])
        except requests.RequestException:
            pass
        time.sleep(1 + a)
    return None


def domain_total_bytes(urls, sample=4):
    """Estimate a domain's total compressed bytes by HEADing a few shards."""
    idxs = sorted({0, len(urls) // 2, len(urls) - 1} |
                  {min(i, len(urls) - 1) for i in range(sample)})
    sizes = [s for s in (head_bytes(urls[i]) for i in idxs) if s]
    if not sizes:
        return None, 0
    return int(sum(sizes) / len(sizes)) * len(urls), len(sizes)


# ----------------------------------------------------------------- workers


def _init_worker(tokenizer_name):
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # we parallelise by process
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_name)


def _iter_texts(url, chunk=1 << 20):
    with requests.get(url, stream=True, timeout=600, headers=HEADERS) as r:
        r.raise_for_status()
        raw = io.BufferedReader(gzip.GzipFile(fileobj=r.raw), buffer_size=chunk)
        for line in raw:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("text")
            if t:
                yield t


def _tokenize_shard(task):
    """One shard -> one uint16 part file. Returns stats (no token data in the result)."""
    url, out_path, batch, cap = task
    eos = _TOK.eos_token_id
    total, nbytes, docs = 0, 0, 0
    t0 = time.time()
    try:
        with open(out_path, "wb") as fh:
            buf = []
            for text in _iter_texts(url):
                buf.append(text)
                nbytes += len(text.encode("utf-8", "ignore"))
                docs += 1
                if len(buf) >= batch:
                    ids = _TOK(buf, add_special_tokens=False)["input_ids"]
                    flat = [t for s in ids for t in (s + [eos])]
                    np.asarray(flat, dtype=np.uint16).tofile(fh)
                    total += len(flat)
                    buf = []
                    if cap and total >= cap:
                        break
            if buf:
                ids = _TOK(buf, add_special_tokens=False)["input_ids"]
                flat = [t for s in ids for t in (s + [eos])]
                np.asarray(flat, dtype=np.uint16).tofile(fh)
                total += len(flat)
    except Exception as e:                              # noqa: BLE001
        return {"url": url, "path": out_path, "error": repr(e), "tokens": 0,
                "bytes": 0, "docs": 0, "elapsed_s": round(time.time() - t0)}
    return {"url": url, "path": out_path, "tokens": total, "bytes": nbytes,
            "docs": docs, "elapsed_s": round(time.time() - t0)}


# ----------------------------------------------------------------- per domain


def est_shards_needed(urls, target, avg_bytes):
    """How many shards to farm out, from a conservative tokens/byte prior."""
    if not avg_bytes:
        return len(urls)
    per_shard = (avg_bytes / 2 ** 30) * TOK_PER_GB_GZ
    n = int(target / max(per_shard, 1)) + 1
    return max(1, min(len(urls), n))


def build_domain(dom, urls, target, outdir, nproc, avg_bytes, batch=512):
    npy = os.path.join(outdir, f"{dom}.npy")
    state_p = os.path.join(outdir, f"{dom}.state.json")
    if os.path.exists(state_p):
        st = json.load(open(state_p))
        if st.get("tokens", 0) >= target or st.get("exhausted"):
            print(f"  {dom}: have {st['tokens']:,} tokens, skipping")
            return st

    parts_dir = os.path.join(outdir, f"_parts_{dom}")
    os.makedirs(parts_dir, exist_ok=True)
    t0 = time.time()
    collected, results, used = 0, [], 0
    pending = list(urls)

    # Farm out shards in rounds until target reached or shards exhausted.
    while collected < target and pending:
        n = est_shards_needed(pending, target - collected, avg_bytes)
        wave, pending = pending[:n], pending[n:]
        tasks = [(u, os.path.join(parts_dir, f"{used + i:05d}.bin"), batch, None)
                 for i, u in enumerate(wave)]
        used += len(wave)
        print(f"  {dom}: dispatching {len(wave)} shard(s) across {nproc} procs "
              f"({collected:,}/{target:,} tokens so far)", flush=True)
        with mp.Pool(nproc, initializer=_init_worker, initargs=(TOKENIZER,)) as pool:
            for res in pool.imap_unordered(_tokenize_shard, tasks):
                if res.get("error"):
                    print(f"    !! {os.path.basename(res['url'])}: {res['error']}")
                    continue
                results.append(res)
                collected += res["tokens"]
                print(f"    + {os.path.basename(res['url'])} "
                      f"{res['tokens']:,} tok in {res['elapsed_s']}s "
                      f"(total {collected:,})", flush=True)
        if collected < target and not pending:
            print(f"  {dom}: POOL EXHAUSTED at {collected:,} tokens "
                  f"(all {len(urls)} shards used) -- this is all of it that exists")

    # Concatenate parts in deterministic shard order, truncating at target.
    # Streamed in chunks into a memmapped .npy: a 6B-token domain is 12 GB, which
    # must never be resident in RAM.
    results.sort(key=lambda r: r["path"])
    avail = sum(os.path.getsize(r["path"]) // 2 for r in results)
    total_len = int(min(target, avail))
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint16,
                                    shape=(total_len,))
    CHUNK = 1 << 26                                   # 64M tokens = 128 MB
    written = 0
    for r in results:
        if written >= total_len:
            break
        with open(r["path"], "rb") as fh:
            while written < total_len:
                want = min(CHUNK, total_len - written)
                chunk = np.fromfile(fh, dtype=np.uint16, count=want)
                if chunk.size == 0:
                    break
                arr[written: written + chunk.size] = chunk
                written += chunk.size
    arr.flush()
    del arr
    for r in results:
        try:
            os.remove(r["path"])
        except OSError:
            pass
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass

    tot_bytes = sum(r["bytes"] for r in results)
    tot_tok = sum(r["tokens"] for r in results)
    st = {"domain": dom, "tokens": int(written), "shards_used": used,
          "shards_available": len(urls),
          "exhausted": (used >= len(urls) and written < target),
          "utf8_bytes_read": tot_bytes, "docs_read": sum(r["docs"] for r in results),
          "tokens_per_byte": (tot_tok / tot_bytes) if tot_bytes else None,
          "elapsed_s": round(time.time() - t0), "path": f"{dom}.npy"}
    json.dump(st, open(state_p, "w"), indent=2)
    print(f"  {dom}: {written:,} tokens from {used}/{len(urls)} shards "
          f"in {st['elapsed_s']}s")
    return st


# ----------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dolma_domains")
    ap.add_argument("--tokens-per-domain", type=int, default=6_000_000_000,
                    help="target per domain; small domains (wiki, books) self-cap below "
                         "this because that is all that exists")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--merge-cc", action="store_true",
                    help="merge cc_en_head/middle/tail into one 'cc_en' domain (k=7)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--batch", type=int, default=512, help="docs per tokenizer call")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    urls = fetch_manifest()
    by_dom = group_by_domain(urls, merge_cc=args.merge_cc)
    doms = args.domains or sorted(by_dom)
    print(f"manifest: {len(urls)} shards, {len(by_dom)} domains -> building {doms}")
    print(f"procs={args.procs}  target={args.tokens_per_domain:,} tokens/domain\n")

    print("sizing domains (HEAD only)...")
    totals, avg = {}, {}
    for d in doms:
        tb, n = domain_total_bytes(by_dom[d])
        totals[d] = tb
        avg[d] = (tb / len(by_dom[d])) if tb else None
        est = (tb or 0) / 2 ** 30 * TOK_PER_GB_GZ
        print(f"  {d:14} {len(by_dom[d]):>5} shards  ~{(tb or 0)/2**30:8.1f} GB  "
              f"~{est/1e9:6.0f}B tokens available  ({n} HEADs)")

    if args.estimate_only:
        json.dump({"compressed_bytes_est": totals,
                   "shards": {d: len(by_dom[d]) for d in doms},
                   "tokens_available_est": {d: (totals[d] or 0) / 2**30 * TOK_PER_GB_GZ
                                            for d in doms}},
                  open(os.path.join(args.out, "size_estimate.json"), "w"), indent=2)
        print("\nwrote size_estimate.json (nothing downloaded)")
        return

    stats = {}
    for d in doms:
        print(f"\n[{d}]")
        stats[d] = build_domain(d, by_dom[d], args.tokens_per_domain, args.out,
                                args.procs, avg[d], batch=args.batch)

    # natural weights: measured tokens/byte scaled to each domain's full byte count
    est_tokens = {}
    for d, st in stats.items():
        tpb, tb = st.get("tokens_per_byte"), totals.get(d)
        est_tokens[d] = int(tpb * tb) if (tpb and tb) else None
    known = {d: v for d, v in est_tokens.items() if v}
    tot = sum(known.values()) or 1
    natural = {d: v / tot for d, v in known.items()}

    json.dump({
        "dolma_version": "v1_5", "tokenizer": TOKENIZER,
        "target_tokens_per_domain": args.tokens_per_domain,
        "domains": stats, "compressed_bytes_est": totals,
        "full_corpus_tokens_est": est_tokens, "natural_weights": natural,
        "natural_weights_note": (
            "AI2 published no per-domain token table for Dolma v1.5. Derived: measured "
            "tokens-per-utf8-byte on shards actually read, scaled by each domain's total "
            "compressed bytes (HEAD-sampled). Estimate, not official."),
        "collected_tokens_total": sum(s["tokens"] for s in stats.values()),
    }, open(os.path.join(args.out, "domain_manifest.json"), "w"), indent=2)

    print("\n=== natural weighting (measured) ===")
    for d in sorted(natural, key=natural.get, reverse=True):
        s = stats[d]
        flag = "   [ALL THAT EXISTS]" if s.get("exhausted") else ""
        print(f"  {d:14} {natural[d]*100:6.2f}%   pool={s['tokens']:>14,} tok{flag}")
    total = sum(s["tokens"] for s in stats.values())
    print(f"\ntotal: {total:,} tokens  ({total*2/2**30:.0f} GB on disk) -> {args.out}/")


if __name__ == "__main__":
    mp.freeze_support()
    main()
