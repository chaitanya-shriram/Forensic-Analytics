"""Investigator explanations: attention (inner + outer level), gradient x input feature
attribution and plain-language reasons derived from behavioural features."""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .data import World
from .features import FEATURE_LABELS, feature_names
from .graph import NODE_DEVICE, NODE_GHOST, NODE_MERCHANT, NODE_OWN, GraphView, extract_subgraphs
from .privacy import pseudonym

# relation of an edge *into the root*, phrased from the root account's point of view
INNER_REL = {0: "received from", 1: "sent to", 2: "uses device", 3: "uses device", 4: "paid merchant", 5: "paid merchant"}
OUTER_REL = {0: "received from", 1: "sent to", 2: "shares device with"}


def describe_node(world: World, view: GraphView, idx: int) -> dict:
    nt = view.node_type()[idx]
    if nt in (NODE_OWN, NODE_GHOST):
        a = int(view.acc_ids[idx])
        return {"kind": "account", "id": a, "name": world.account_name(a), "bank": int(world.acc_bank[a]),
                "external": bool(nt == NODE_GHOST), "member": bool(world.acc_bank[a] < world.n_banks), "pseudonym": pseudonym("acc", a)}
    if nt == NODE_DEVICE:
        d = int(view.dev_ids[idx - view.n_acc])
        return {"kind": "device", "id": d, "name": f"device {pseudonym('dev', d)[:8]}"}
    m = int(view.mer_ids[idx - view.n_acc - len(view.dev_ids)])
    return {"kind": "merchant", "id": m, "name": world.merchant_names[m], "high_risk": bool(world.merchant_risk[m])}


def reasons(world: World, X_raw: np.ndarray, a: int) -> list:
    names = feature_names(world)
    f = dict(zip(names, X_raw[a]))
    member = world.is_member
    pct = lambda name: float((X_raw[member, names.index(name)] < f[name]).mean())
    # share of member accounts at least this extreme: a reason is only worth showing if it is rare
    rare = lambda name, q=0.15: float((X_raw[member, names.index(name)] >= f[name]).mean()) < q
    out = []
    if f["pass_through_ratio"] > 0.8 and f["log_in_amount"] > 7:
        out.append(f"Forwards {f['pass_through_ratio']:.0%} of received funds (pass-through / mule behaviour).")
    hold = np.expm1(f["log_holding_minutes"])
    if hold < 240:
        out.append(f"Median time between receiving and sending funds is {hold:.0f} min (rapid layering).")
    if f["structuring_ratio"] > 0.15:
        out.append(f"{f['structuring_ratio']:.0%} of transfers are just below the 10,000 reporting threshold (structuring).")
    if (f["cross_bank_in_ratio"] > 0.6 and rare("cross_bank_in_ratio")) or (f["cross_bank_out_ratio"] > 0.6 and rare("cross_bank_out_ratio")):
        out.append(f"Cross-bank flow: {f['cross_bank_in_ratio']:.0%} of inflows and {f['cross_bank_out_ratio']:.0%} of outflows involve other banks.")
    if "log_device_sharing" in f:
        if f["log_device_sharing"] > np.log1p(2):
            out.append(f"Device shared with {np.expm1(f['log_device_sharing']):.0f} other accounts across {f['device_bank_span']:.0f} bank(s).")
        if f["risky_merchant_ratio"] > 0.2:
            out.append(f"{f['risky_merchant_ratio']:.0%} of outgoing value goes to high-risk merchants (crypto / gift cards / remittance).")
        age = np.expm1(f["log_account_age_days"])
        if age < 90:
            out.append(f"Account opened only {age:.0f} days ago.")
    if "share_bitcoin" in f:
        risky = f["share_bitcoin"] + f["share_cash"]
        if risky > 0.4:
            out.append(f"{risky:.0%} of transfers use Bitcoin or cash rails (hard to trace).")
        if f["share_ach"] > 0.6 and pct("log_out_counterparties") > 0.9:
            out.append(f"ACH-heavy ({f['share_ach']:.0%}) payouts to many receivers, a typical layering rail.")
        if f["cross_currency_ratio"] > 0.3:
            out.append(f"{f['cross_currency_ratio']:.0%} of transfers convert currency.")
        if f["external_ratio"] > 0.7 and rare("external_ratio"):
            out.append(f"{f['external_ratio']:.0%} of counterparties bank outside the consortium.")
        if np.expm1(f["fan_in_burst"]) >= 4:
            out.append(f"Received from {np.expm1(f['fan_in_burst']):.0f} different senders within one day (fan-in burst).")
        if np.expm1(f["fan_out_burst"]) >= 4:
            out.append(f"Paid {np.expm1(f['fan_out_burst']):.0f} different receivers within one day (fan-out burst).")
        if pct("log_max_amount") > 0.98:
            out.append(f"Largest transfer ${np.expm1(f['log_max_amount']):,.0f} is in the top 2% of the consortium.")
    if pct("log_in_counterparties") > 0.97:
        out.append(f"Receives from {np.expm1(f['log_in_counterparties']):.0f} distinct senders (top 3% fan-in).")
    if pct("log_out_counterparties") > 0.97:
        out.append(f"Sends to {np.expm1(f['log_out_counterparties']):.0f} distinct receivers (top 3% fan-out).")
    return out


def explain_account(engine, a: int, top_k: int = 8) -> dict:
    world: World = engine.world
    b = int(world.acc_bank[a])
    view: GraphView = engine.prep.banks[b]
    pos = int(np.searchsorted(view.acc_ids[: view.n_own], a))
    model = engine.model
    model.eval()
    ge, gm = engine.ghost_inputs(b)

    # --- inner (nested) level: which nodes of the rooted subgraph mattered
    sub = extract_subgraphs(view, [pos]).to(engine.device)
    x = view.x_acc.clone().requires_grad_(True)
    v2 = dataclasses.replace(view, x_acc=x)
    z, inner = model.encode_subgraphs(v2, sub, ge, gm, return_attention=True)
    gate = inner[-1].detach().cpu().numpy()
    att_last = inner[-2].detach().cpu().numpy()
    node_idx = sub.node_idx.cpu().numpy()
    ei = sub.edge_index.cpu().numpy()
    et = sub.edge_type.cpu().numpy()
    into_root = np.flatnonzero(ei[1] == 0)
    inner_items = []
    for e in into_root[np.argsort(-att_last[into_root])][:top_k]:
        n = describe_node(world, view, int(node_idx[ei[0, e]]))
        inner_items.append({**n, "relation": INNER_REL[int(et[e])], "weight": float(att_last[e])})
    pool_items = []
    for k in np.argsort(-gate)[:top_k]:
        n = describe_node(world, view, int(node_idx[k]))
        pool_items.append({**n, "hop": int(sub.dist[k]), "weight": float(gate[k])})

    # --- outer (cross-bank) level + gradient attribution in one pass over the full bank view
    z_all = engine.z_cache[b].clone()
    z_all[pos] = z[0]
    logits, h, outer = model.outer_forward(v2, z_all, ge, gm, return_attention=True)
    logit = logits[pos]
    logit.backward()
    grad = x.grad[pos].detach().cpu().numpy()
    xin = view.x_acc[pos].detach().cpu().numpy()
    contrib = grad * xin
    names = feature_names(world)
    feats = sorted(({"feature": names[i], "label": FEATURE_LABELS[names[i]],
                     "value": float(engine.X_raw[a, i]), "contribution": float(contrib[i])}
                    for i in range(len(xin))), key=lambda d: -abs(d["contribution"]))
    oa = outer[-1].detach().cpu().numpy()
    oei = view.acc_edge_index.cpu().numpy()
    oet = view.acc_edge_type.cpu().numpy()
    into = np.flatnonzero(oei[1] == pos)
    agg: dict[int, dict] = {}
    for e in into:
        n = int(oei[0, e])
        d = agg.setdefault(n, {**describe_node(world, view, n), "relations": set(), "weight": 0.0})
        d["relations"].add(OUTER_REL[int(oet[e])])
        d["weight"] += float(oa[e])
    outer_items = sorted(agg.values(), key=lambda d: -d["weight"])[:top_k]
    for d in outer_items:
        d["relations"] = sorted(d["relations"])
        d["score"] = float(engine.scores[d["id"]])
        d["bank_name"] = world.bank_name(d["bank"])
    # full rooted subgraph (for the step-by-step nested-GNN view); forward-direction edges only
    nt = view.node_type()
    sub_nodes = []
    for k, vi in enumerate(node_idx):
        d = describe_node(world, view, int(vi))
        if d["kind"] == "account" and world.acc_bank[d["id"]] < world.n_banks:
            d["score"] = float(engine.scores[d["id"]])
        sub_nodes.append({**d, "hop": int(sub.dist[k]), "pool": float(gate[k]), "label": int(world.acc_label[d["id"]]) if d["kind"] == "account" else 0})
    kinds = {0: "transfer", 2: "device", 4: "merchant"}
    att_first = inner[0].detach().cpu().numpy()
    sub_edges = [{"source": int(ei[0, e]), "target": int(ei[1, e]), "kind": kinds[int(et[e])],
                  "attention": float(att_last[e]), "attention_l1": float(att_first[e])}
                 for e in range(ei.shape[1]) if int(et[e]) in kinds]
    return {
        "subgraph": {"nodes": sub_nodes, "edges": sub_edges, "root": 0, "node_types": int(len(set(nt[node_idx].tolist())))},
        "z": [float(v) for v in z[0].detach().cpu().numpy()],
        "p_acc": float(engine.acc_prob[a]) if hasattr(engine, "acc_prob") else float(torch.sigmoid(logit).item()),
        "tx_max": float(engine.tx_max[a]) if hasattr(engine, "tx_max") else None,
        "score": float(engine.scores[a]),
        "logit": float(logit.detach()),
        "reasons": reasons(world, engine.X_raw, a),
        "features": feats,
        "inner_attention": inner_items,
        "subgraph_pooling": pool_items,
        "outer_attention": outer_items,
        "subgraph_size": int(len(node_idx)),
    }
