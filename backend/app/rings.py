"""Consortium-level fraud ring assembly.

Each bank reports only pseudonymised ids of accounts it scores above threshold, plus the
pseudonymised counterparties / devices of those accounts. The coordinator links flagged
accounts that transact with each other or share a device and returns connected components
as candidate rings. No unflagged customer data leaves a bank.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .data import TX_TRANSFER, World
from .privacy import pseudonym

PATTERN_LABELS = {
    "layering_chain": "Layering chain", "fan_in_fan_out": "Fan-in / fan-out mule network",
    "round_tripping": "Round-tripping cycle", "device_farm": "Device farm / synthetic identities",
}


def flagged_graph(world: World, scores: np.ndarray, threshold: float) -> nx.Graph:
    flagged = np.flatnonzero(scores >= threshold)
    fset = set(flagged.tolist())
    G = nx.Graph()
    for a in flagged:
        G.add_node(int(a))
    t = world.tx
    m = t.view("kind") == TX_TRANSFER
    src, dst, amt = t.view("src")[m], t.view("dst")[m], t.view("amount")[m]
    fm = np.zeros(world.n_accounts, bool)
    fm[flagged] = True
    sel = fm[src] & fm[dst] & (src != dst)
    for s, d, a in zip(src[sel], dst[sel], amt[sel]):
        e = G.get_edge_data(int(s), int(d))
        if e is None:
            G.add_edge(int(s), int(d), amount=float(a), count=1, device=False)
        else:
            e["amount"] += float(a)
            e["count"] += 1
    # flagged accounts that move money through the same intermediary (often an account at a
    # non-member institution the consortium only sees via its own ledgers)
    via = fm[src] ^ fm[dst]
    vs = np.where(fm[src[via]], src[via], dst[via])
    vc = np.where(fm[src[via]], dst[via], src[via])
    va = amt[via]
    hub: dict[int, dict] = {}
    for f, c, a in zip(vs.tolist(), vc.tolist(), va.tolist()):
        hub.setdefault(c, {}).setdefault(f, 0.0)
        hub[c][f] += a
    for c, fs in hub.items():
        if 1 < len(fs) <= 12:
            ks = list(fs)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    w_ = min(fs[ks[i]], fs[ks[j]])
                    if not G.has_edge(ks[i], ks[j]):
                        G.add_edge(ks[i], ks[j], amount=0.0, count=0, device=False, via=0.0)
                    e = G[ks[i]][ks[j]]
                    e["via"] = e.get("via", 0.0) + w_
    dev: dict[str, list] = {}
    for a in flagged:
        for d in world.acc_devices[a]:
            dev.setdefault(pseudonym("dev", d), []).append(int(a))
    for us in dev.values():
        if 1 < len(us) <= 30:
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    if G.has_edge(us[i], us[j]):
                        G[us[i]][us[j]]["device"] = True
                    else:
                        G.add_edge(us[i], us[j], amount=0.0, count=0, device=True)
    return G


_pattern_cache: dict = {}


def classify_pattern(world: World, members: list) -> str:
    """Motif-based typology of a candidate ring (for investigators; not used for scoring)."""
    key = (id(world), tuple(members))
    if key not in _pattern_cache:
        if len(_pattern_cache) > 5000:
            _pattern_cache.clear()
        _pattern_cache[key] = _classify(world, members)
    return _pattern_cache[key]


def _classify(world: World, members: list) -> str:
    t = world.tx
    m = (t.view("kind") == TX_TRANSFER) & np.isin(t.view("src"), members) & np.isin(t.view("dst"), members)
    D = nx.DiGraph()
    D.add_nodes_from(members)
    D.add_edges_from(zip(t.view("src")[m].tolist(), t.view("dst")[m].tolist()))
    dev: dict[int, int] = {}
    for a in members:
        for d in world.acc_devices[a]:
            dev[d] = dev.get(d, 0) + 1
    n = len(members)
    device_share = max(dev.values(), default=0) / n
    on_cycle = sum(len(c) for c in nx.strongly_connected_components(D) if len(c) >= 3) / n
    indeg = sorted((D.in_degree(a) for a in members), reverse=True)
    outdeg = sorted((D.out_degree(a) for a in members), reverse=True)
    if device_share >= 0.4:
        return "device_farm"
    if on_cycle >= 0.6:
        return "round_tripping"
    if indeg[0] >= 3 and outdeg[0] >= 3 or sum(d >= 3 for d in indeg) >= 2:
        return "fan_in_fan_out"
    return "layering_chain"


MIN_LINK_AMOUNT = 1000.0


def ring_candidates(G: nx.Graph, min_size: int, seed: int = 0) -> list:
    """Keep strong links (material money flow, repeated transfers or shared devices), then split
    large components with weighted Louvain so incidental contacts don't merge distinct rings."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        via = d.get("via", 0.0)
        if d["device"] or d["amount"] >= MIN_LINK_AMOUNT or d["count"] >= 3 or via >= MIN_LINK_AMOUNT:
            H.add_edge(u, v, weight=np.log1p(d["amount"]) + 0.5 * np.log1p(via) + 3.0 * d["device"] + d["count"])
    out = []
    for comp in nx.connected_components(H):
        if len(comp) < min_size:
            continue
        if len(comp) <= 14:  # a single ring rarely exceeds this; only split merged components
            out.append(comp)
            continue
        for c in nx.community.louvain_communities(H.subgraph(comp), weight="weight", resolution=3.0, seed=seed):
            if len(c) >= min_size:
                out.append(c)
    return out


def detect_rings(world: World, scores: np.ndarray, threshold: float, min_size: int = 3) -> list:
    G = flagged_graph(world, scores, threshold)
    rings = []
    for comp in ring_candidates(G, min_size):
        members = sorted(comp)
        sub = G.subgraph(members)
        flow = sum(d["amount"] for _, _, d in sub.edges(data=True))
        truth = world.acc_ring[members]
        true_ids, counts = np.unique(truth[truth >= 0], return_counts=True)
        best = int(true_ids[counts.argmax()]) if len(true_ids) else -1
        pattern = classify_pattern(world, members)
        rings.append({
            "members": [int(a) for a in members],
            "size": len(members),
            "banks": sorted({int(world.acc_bank[a]) for a in members}),
            "total_flow": round(flow, 2),
            "mean_score": float(np.mean(scores[members])),
            "max_score": float(np.max(scores[members])),
            "n_links": sub.number_of_edges(),
            "device_links": sum(1 for _, _, d in sub.edges(data=True) if d["device"]),
            "pattern": pattern,
            "pattern_label": PATTERN_LABELS[pattern],
            "matched_true_ring": best,
            "precision": float((truth >= 0).mean()),
        })
    rings.sort(key=lambda r: (-r["mean_score"] * np.log1p(r["size"])))
    for i, r in enumerate(rings):
        r["id"] = i
        r["pseudonym"] = pseudonym("ring", hash(tuple(r["members"])) & 0xFFFFFFFF)[:10]
    return rings


def ring_recall(world: World, detected: list, eval_rings=None, min_overlap: float = 0.5) -> float:
    """Share of true rings for which a detected ring covers >= min_overlap of its members."""
    truth = [r for r in world.rings if eval_rings is None or r["id"] in eval_rings]
    if not truth:
        return float("nan")
    hit = 0
    for r in truth:
        mem = set(r["members"])
        best = max((len(mem & set(d["members"])) / len(mem) for d in detected), default=0)
        hit += best >= min_overlap
    return hit / len(truth)


TX_RING_QUANTILE = 0.9995  # rings are built from the riskiest 0.05% of transactions


def tx_ring_threshold(tx_risk: np.ndarray, floor: float = 0.0) -> float:
    r = tx_risk[~np.isnan(tx_risk)]
    return max(float(np.quantile(r, TX_RING_QUANTILE)), floor) if len(r) else 1.0


def detect_tx_rings(world: World, tx_risk: np.ndarray, threshold: float, min_size: int = 3) -> list:
    """Rings as connected groups of high-risk *transactions* (the granularity at which real AML
    labels exist). Accounts at non-member institutions appear as intermediaries; the coordinator
    only ever sees pseudonyms of the accounts on flagged transactions."""
    t = world.tx
    r = np.nan_to_num(tx_risk[: t.n])
    sel = np.flatnonzero((r >= threshold) & (t.view("dst") >= 0))
    G = nx.Graph()
    for i in sel:
        s, d = int(t.src[i]), int(t.dst[i])
        e = G.get_edge_data(s, d)
        if e is None:
            G.add_edge(s, d, amount=float(t.amount[i]), count=1, risk=[float(r[i])], label=[int(t.label[i])], tx=[int(i)])
        else:
            e["amount"] += float(t.amount[i]); e["count"] += 1
            e["risk"].append(float(r[i])); e["label"].append(int(t.label[i])); e["tx"].append(int(i))
    truth = [(k, set(x.get("all_members", x["members"]))) for k, x in enumerate(world.rings)]
    rings = []
    for comp in nx.connected_components(G):
        if len(comp) < min_size:
            continue
        members = sorted(comp)
        sub = G.subgraph(members)
        risks = [x for _, _, d in sub.edges(data=True) for x in d["risk"]]
        labels = [x for _, _, d in sub.edges(data=True) for x in d["label"]]
        cs = set(members)
        best, best_ov = -1, 0.0
        for k, ts in truth:
            ov = len(ts & cs) / len(ts) if ts else 0
            if ov > best_ov:
                best, best_ov = k, ov
        rings.append({
            "members": [int(a) for a in members],
            "size": len(members),
            "banks": sorted({int(world.acc_bank[a]) for a in members}),
            "total_flow": round(sum(d["amount"] for _, _, d in sub.edges(data=True)), 2),
            "mean_score": float(np.mean(risks)), "max_score": float(np.max(risks)),
            "n_links": sub.number_of_edges(), "device_links": 0,
            "pattern": classify_pattern(world, members), "matched_true_ring": best if best_ov >= 0.2 else -1,
            "precision": float(np.mean(labels)), "n_tx": len(risks),
            "member_accounts": int(sum(world.acc_bank[a] < world.n_banks for a in members)),
        })
        rings[-1]["pattern_label"] = PATTERN_LABELS.get(rings[-1]["pattern"], rings[-1]["pattern"])
    rings.sort(key=lambda x: -(x["mean_score"] * np.log1p(x["n_tx"])))
    for i, x in enumerate(rings):
        x["id"] = i
        x["pseudonym"] = pseudonym("ring", hash(tuple(x["members"])) & 0xFFFFFFFF)[:10]
    return rings


def tx_ring_recall(world: World, detected: list, eval_rings, min_overlap: float = 0.5) -> float:
    truth = [set(world.rings[k].get("all_members", world.rings[k]["members"])) for k in eval_rings]
    if not truth:
        return float("nan")
    return float(np.mean([max((len(t & set(d["members"])) / len(t) for d in detected), default=0) >= min_overlap for t in truth]))
