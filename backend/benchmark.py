"""Multi-seed benchmark of all five models.

Each seed re-draws the ring-disjoint train/val/test split and the model initialisation, which
matters on the real data where the test set holds only a few dozen laundering accounts.

Usage: python benchmark.py [n_seeds] [ibm_aml|synthetic]
       -> prints mean ± std, writes artifacts/benchmark_<dataset>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from app.data import generate_world
from app.train import MODEL_INFO, evaluate, exchange_embeddings, predict, prepare, train_central, train_federated

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dataset = sys.argv[2] if len(sys.argv) > 2 else "ibm_aml"
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dataset == "ibm_aml":
    from app.aml import load_ibm_aml
    world, epochs, rounds = load_ibm_aml(), 200, 40
else:
    world, epochs, rounds = generate_world(seed=7), 150, 25
METRICS = ("auc", "ap", "f1", "tx_auc", "tx_ap", "ring_recall")
rows = {k: [] for k in MODEL_INFO}
for seed in range(n):
    prep = prepare(world, seed=seed, device=dev)
    for kind in ("mlp", "flat_gnn", "nested_central"):
        m, _ = train_central(kind, prep, epochs=epochs, seed=seed)
        rows[kind].append(evaluate(prep, predict(m, prep, kind)))
    for kind, ex in (("nested_local", False), ("nested_federated", True)):
        m, _, _ = train_federated(prep, rounds=rounds, exchange=ex, seed=seed)
        rows[kind].append(evaluate(prep, predict(m, prep, kind, exchange_embeddings(m, prep) if ex else None)))
    print(f"seed {seed} done", flush=True)


def stat(rs, m):
    v = [r[m] for r in rs if r.get(m) is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


summary = {k: {m: stat(rs, m) for m in METRICS} for k, rs in rows.items()}
base = float(np.mean([r["base_rate"] for r in rows["mlp"]]))
for k, s in summary.items():
    print(f"{k:18s} " + "  ".join(f"{m} {mu:.3f}±{sd:.3f}" for m, (mu, sd) in s.items() if mu is not None))
print(f"account base rate {base:.4f}")
Path("artifacts").mkdir(exist_ok=True)
Path(f"artifacts/benchmark_{dataset}.json").write_text(json.dumps({"seeds": n, "summary": summary, "base_rate": base, "runs": rows}, indent=1))
