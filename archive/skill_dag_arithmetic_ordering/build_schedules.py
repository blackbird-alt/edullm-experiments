"""
Schedule builder for the skill-DAG sequencing experiment.

Produces the ONLY thing the two arms differ in: the order records are seen.
- 3 topological schedules (valid linearizations of dag.json's edges, seeded
  Kahn tie-breaking; ALL records of every prerequisite node appear before ANY
  record of the dependent node).
- 3 random schedules (full shuffles of the same records).

Note: scheduling uses the EDGES, not the descriptive `tiers` field in dag.json
(the tiers list puts ADD and MUL in one tier despite the ADD->MUL edge; edges
are authoritative).

Output: schedules/<name>.idx (one train.jsonl line-index per line) + manifest.json.
Every topological schedule is verified against the constraint before writing.
"""
import json, os, random
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "skill_dag_dataset")
OUT = os.path.join(HERE, "schedules")
os.makedirs(OUT, exist_ok=True)

TOPO_SEEDS = [101, 102, 103]
RAND_SEEDS = [201, 202, 203]

# ---- load ----
with open(os.path.join(DATA, "dag.json")) as f:
    dag = json.load(f)
edges = [tuple(e) for e in dag["edges"]]
nodes = list(dag["nodes"].keys())

records_by_skill = defaultdict(list)  # skill -> [line indices]
with open(os.path.join(DATA, "train.jsonl")) as f:
    for i, line in enumerate(f):
        records_by_skill[json.loads(line)["skill"]].append(i)
n_total = sum(len(v) for v in records_by_skill.values())


def topo_order(seed):
    """Kahn's algorithm with seeded random tie-breaking -> one valid node linearization."""
    rng = random.Random(seed)
    indeg = {n: 0 for n in nodes}
    children = defaultdict(list)
    for a, b in edges:
        indeg[b] += 1
        children[a].append(b)
    ready = sorted([n for n in nodes if indeg[n] == 0])
    order = []
    while ready:
        rng.shuffle(ready)
        n = ready.pop()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    assert len(order) == len(nodes), "cycle in DAG?"
    return order


def verify_topological(stream, node_order_name):
    """Hard check: max position of every prereq's records < min position of dependent's."""
    pos = {}
    with open(os.path.join(DATA, "train.jsonl")) as f:
        skills = [json.loads(l)["skill"] for l in f]
    first, last = {}, {}
    for p, idx in enumerate(stream):
        sk = skills[idx]
        first.setdefault(sk, p)
        last[sk] = p
    for a, b in edges:
        assert last[a] < first[b], f"{node_order_name}: edge {a}->{b} violated (last {a}={last[a]}, first {b}={first[b]})"


manifest = {"dataset_version": dag["version"], "n_records": n_total, "schedules": {}}

# topological arms
for seed in TOPO_SEEDS:
    order = topo_order(seed)
    rng = random.Random(seed * 7 + 1)
    stream = []
    for node in order:
        recs = list(records_by_skill[node])
        rng.shuffle(recs)          # within-node order randomized per seed
        stream.extend(recs)
    name = f"topo_{seed}"
    verify_topological(stream, name)
    with open(os.path.join(OUT, name + ".idx"), "w") as f:
        f.write("\n".join(map(str, stream)))
    manifest["schedules"][name] = {"type": "topological", "seed": seed, "node_order": order}
    print(f"{name}: node order = {' -> '.join(order)}  ({len(stream)} records, constraint verified)")

# random arms
all_records = [i for v in records_by_skill.values() for i in v]
for seed in RAND_SEEDS:
    rng = random.Random(seed)
    stream = list(all_records)
    rng.shuffle(stream)
    name = f"random_{seed}"
    with open(os.path.join(OUT, name + ".idx"), "w") as f:
        f.write("\n".join(map(str, stream)))
    manifest["schedules"][name] = {"type": "random", "seed": seed}
    print(f"{name}: full shuffle ({len(stream)} records)")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest written; all schedules contain identical record sets, order is the only difference")
