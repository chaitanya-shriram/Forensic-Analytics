"""Data preparation, centralised / federated training and evaluation."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .data import World
from .features import (DEVICE_FEATURES, MERCHANT_FEATURES, Normalizer, account_features, device_features,
                       device_stats, feature_names, tx_frame)
from .graph import GraphView, SubgraphBatch, build_view, extract_chunks, subs_to
from .model import FeatureMLP, NestedHGNN
from .privacy import clip, flat, gaussian_noise, psi_device_counts, rdp_epsilon, secure_aggregate, unflat
from .rings import detect_rings, detect_tx_rings, ring_recall, tx_ring_recall, tx_ring_threshold

POS_WEIGHT_CAP = 30.0

MODEL_INFO = {
    "mlp": "Feature MLP (per-transaction baseline)",
    "flat_gnn": "Flat heterogeneous GNN (central, no nesting)",
    "nested_local": "Nested GNN, federated, no cross-bank exchange",
    "nested_federated": "Nested GNN, federated + cross-bank embedding exchange (proposed)",
    "nested_central": "Nested GNN, centralised pooled data (upper bound)",
}


@dataclass
class Prepared:
    world: World
    X_raw: np.ndarray
    X: np.ndarray
    norm: Normalizer
    X_dev: np.ndarray
    dev_stats: tuple
    central: GraphView
    central_sub: SubgraphBatch
    banks: list
    bank_subs: list
    y: np.ndarray
    split: np.ndarray          # 0 train / 1 val / 2 test
    device: str = "cpu"
    timings: dict = field(default_factory=dict)

    @property
    def dims(self):
        return len(feature_names(self.world)), len(DEVICE_FEATURES), len(MERCHANT_FEATURES)


def make_split(world: World, seed: int = 0) -> np.ndarray:
    """0 train / 1 val / 2 test (ring-disjoint), -1 external (not owned), 3 live-only positives
    (launder only after the history cutoff: unknown at training time, used for stream evaluation)."""
    rng = np.random.default_rng(seed)
    split = np.zeros(world.n_accounts, np.int64)
    hist = [r for r in world.rings if not r.get("live") and any(world.acc_label[m] for m in r["members"])]
    # rings that share accounts (common in real data) are split as one group
    import networkx as nx
    G = nx.Graph()
    for i, r in enumerate(hist):
        G.add_node(("r", i))
        for m in r["members"]:
            G.add_edge(("r", i), ("a", m))
    groups = [[i for t, i in c if t == "r"] for c in nx.connected_components(G)]
    groups.sort(key=lambda g: min(g))
    order = rng.permutation(len(groups))
    n_tr, n_va = int(0.6 * len(groups)), int(0.2 * len(groups))
    for k, gi in enumerate(order):
        for i in groups[gi]:
            split[hist[i]["members"]] = 0 if k < n_tr else (1 if k < n_tr + n_va else 2)
    assigned = np.zeros(world.n_accounts, bool)
    for r in hist:
        assigned[r["members"]] = True
    rest = np.flatnonzero(~assigned)
    u = rng.random(len(rest))
    split[rest] = np.where(u < 0.6, 0, np.where(u < 0.8, 1, 2))
    live = getattr(world, "live_pos", None)
    if live is not None:
        split[live] = 3
    split[~world.is_member] = -1
    return split


def transaction_split(world: World, split: np.ndarray):
    """A transaction inherits the latest split of its member endpoints (train < val < test < live),
    so no test or live account's transactions are ever trained on."""
    t = world.tx
    src, dst = t.view("src"), t.view("dst")
    s_src = np.where(world.is_member[src], split[src], -1)
    d = np.maximum(dst, 0)
    s_dst = np.where((dst >= 0) & world.is_member[d], split[d], -1)
    return t.view("label").astype(np.int64).copy(), np.maximum(s_src, s_dst)


def prepare(world: World, seed: int = 0, device: str = "cpu") -> Prepared:
    t0 = time.time()
    ds = device_stats(world)
    if world.n_devices:
        n_acc, n_banks = psi_device_counts(world.acc_devices, world.acc_bank, world.n_devices)
        ds = (n_acc, n_banks, ds[2])
    X_raw = account_features(world, dev_stats=ds)
    norm = Normalizer(X_raw[world.is_member])
    X = norm(X_raw)
    X_dev = device_features(world, ds)
    df = tx_frame(world)
    t1 = time.time()
    central = build_view(world, X, X_dev, None, df)
    banks = [build_view(world, X, X_dev, b, df) for b in range(world.n_banks)]
    t2 = time.time()
    central_sub = extract_chunks(central)
    bank_subs = [extract_chunks(v) for v in banks]
    t3 = time.time()
    p = Prepared(world, X_raw, X, norm, X_dev, ds, central, central_sub, banks, bank_subs,
                 world.acc_label.copy(), make_split(world, seed), device,
                 {"features_s": t1 - t0, "views_s": t2 - t1, "subgraphs_s": t3 - t2})
    p.tx_y, p.tx_split = transaction_split(world, p.split)
    for v in [central, *banks]:
        v.to(device)
    p.central_sub = subs_to(central_sub, device)
    p.bank_subs = [subs_to(s, device) for s in bank_subs]
    return p


def new_model(kind: str, prep: Prepared, hidden: int = 64):
    f_acc, f_dev, f_mer = prep.dims
    if kind == "mlp":
        return FeatureMLP(f_acc, hidden)
    return NestedHGNN(f_acc, f_dev, f_mer, hidden=hidden, nested=(kind != "flat_gnn"))


# ------------------------------------------------------------------ cross-bank embedding exchange
@torch.no_grad()
def exchange_embeddings(model: NestedHGNN, prep: Prepared, noise_std: float = 0.05, clip_norm: float = 1.0,
                        seed: int = 0):
    """Every bank encodes its own accounts with the (shared) inner GNN, clips + noises the
    embeddings and publishes them under pseudonyms. Banks then look up their ghosts."""
    was = model.training
    model.eval()
    H = model.hidden
    pub = torch.zeros(prep.world.n_accounts, H, device=prep.device)
    has = torch.zeros(prep.world.n_accounts, dtype=torch.bool, device=prep.device)
    needed = np.zeros(prep.world.n_accounts, bool)
    for v in prep.banks:
        needed[v.acc_ids[v.n_own:]] = True
    for b, (v, s) in enumerate(zip(prep.banks, prep.bank_subs)):
        z, _ = model.encode_subgraphs(v, s)
        z = z * torch.clamp(clip_norm / (z.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        if noise_std > 0:
            z = z + gaussian_noise(z.shape, noise_std * clip_norm, seed * 31 + b).to(z.device)
        ids = torch.as_tensor(v.acc_ids[: v.n_own], device=prep.device)
        sel = torch.as_tensor(needed[v.acc_ids[: v.n_own]], device=prep.device)
        pub[ids[sel]] = z[sel]
        has[ids[sel]] = True
    model.train(was)
    return pub, has


def ghost_inputs(view: GraphView, pub, has):
    if pub is None or view.n_ghost == 0:
        return None, None
    g = torch.as_tensor(view.acc_ids[view.n_own:], device=pub.device)
    return pub[g], has[g]


# ------------------------------------------------------------------ evaluation
CENTRAL = ("mlp", "flat_gnn", "nested_central")
COMBINE = ("acc", "txmax", "mean")


@torch.no_grad()
def predict(model, prep: Prepared, mode: str, exchange=None) -> dict:
    """Account probabilities ("acc"), transaction probabilities ("tx", over the history store, NaN
    where no view sees the transaction) and each account's highest transaction risk ("txmax")."""
    model.eval()
    N, T = prep.world.n_accounts, prep.world.tx.n
    acc = np.zeros(N)
    tx = np.full(T, np.nan)
    if mode in CENTRAL:
        views = [(prep.central, prep.central_sub, None, None)]
    else:
        pub, has = exchange if exchange is not None else (None, None)
        views = [(v, s, *ghost_inputs(v, pub, has)) for v, s in zip(prep.banks, prep.bank_subs)]
    for v, s, ge, gm in views:
        logits, aux = model(v, s, ge, gm)
        acc[v.acc_ids[: v.n_own]] = torch.sigmoid(logits).cpu().numpy()
        if v.tx_ids is not None and len(v.tx_ids):
            p = torch.sigmoid(model.tx_logits(v, aux["rep"])).cpu().numpy()
            tx[v.tx_ids] = np.fmax(tx[v.tx_ids], p)
    return {"acc": acc, "tx": tx, "txmax": account_tx_max(prep.world, tx)}


def account_tx_max(world: World, tx: np.ndarray) -> np.ndarray:
    t = world.tx
    txmax = np.zeros(world.n_accounts)
    idx = np.flatnonzero(~np.isnan(tx[: t.n]))
    np.maximum.at(txmax, t.src[idx], tx[idx])
    dst = t.dst[idx]
    ok = dst >= 0
    np.maximum.at(txmax, dst[ok], tx[idx][ok])
    txmax[~world.is_member] = 0
    return txmax


def combine(pred: dict, how: str = "acc") -> np.ndarray:
    if how == "txmax":
        return pred["txmax"]
    if how == "mean":
        return 0.5 * (pred["acc"] + pred["txmax"])
    return pred["acc"]


def select_combine(prep: Prepared, pred: dict) -> tuple:
    """Pick how account risk combines account and transaction evidence, by validation AP."""
    va = prep.split == 1
    best = max(COMBINE, key=lambda h: average_precision_score(prep.y[va], combine(pred, h)[va]))
    return best, float(average_precision_score(prep.y[va], combine(pred, best)[va]))


def best_threshold(y, s):
    best, thr = -1, 0.5
    for t in np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 160))):
        f = f1_score(y, s >= t, zero_division=0)
        if f > best:
            best, thr = f, float(t)
    return thr


def evaluate(prep: Prepared, pred: dict, threshold: float | None = None, how: str | None = None) -> dict:
    y, sp = prep.y, prep.split
    va, te = sp == 1, sp == 2
    if how is None:
        how, _ = select_combine(prep, pred)
    scores = combine(pred, how)
    if threshold is None:
        threshold = best_threshold(y[va], scores[va])
    pred_te = scores[te] >= threshold
    rings = prep.world.rings
    test_rings = {int(prep.world.acc_ring[a]) for a in np.flatnonzero(te & (y == 1))}
    test_rings = {r for r in test_rings if r >= 0 and len(rings[r]["members"]) >= 3}  # detectable by the consortium
    masked = scores.copy()
    masked[sp != 2] = 0  # detect rings on test accounts only
    det = detect_rings(prep.world, masked, threshold)
    out = {
        "auc": float(roc_auc_score(y[te], scores[te])),
        "ap": float(average_precision_score(y[te], scores[te])),
        "f1": float(f1_score(y[te], pred_te, zero_division=0)),
        "precision": float(precision_score(y[te], pred_te, zero_division=0)),
        "recall": float(recall_score(y[te], pred_te, zero_division=0)),
        "threshold": float(threshold),
        "ring_recall": float(ring_recall(prep.world, det, test_rings)) if test_rings else None,
        "val_ap": float(average_precision_score(y[va], scores[va])),
        "combine": how,
        "base_rate": float(y[te].mean()),
        "test_positives": int(y[te].sum()),
    }
    if prep.world.source == "ibm_aml":  # rings from high-risk test transactions
        tt = np.where(prep.tx_split == 2, pred["tx"], np.nan)
        live_hist = [k for k, r in enumerate(rings) if not r.get("live")]
        eval_r = [k for k in live_hist if any(sp[a] == 2 for a in rings[k]["members"])
                  and len(rings[k].get("all_members", [])) >= 3]
        det_tx = detect_tx_rings(prep.world, tt, tx_ring_threshold(tt))
        out["ring_recall"] = tx_ring_recall(prep.world, det_tx, eval_r) if eval_r else None
        out["ring_precision"] = float(np.mean([d["precision"] for d in det_tx])) if det_tx else None
        out["rings_detected"] = len(det_tx)
    ty, tsc = prep.tx_y, pred["tx"]
    tva = (prep.tx_split == 1) & ~np.isnan(tsc)
    tte = (prep.tx_split == 2) & ~np.isnan(tsc)
    if ty[tte].sum() and ty[tva].sum():
        tthr = best_threshold(ty[tva], tsc[tva])
        tp = tsc[tte] >= tthr
        out.update({"tx_auc": float(roc_auc_score(ty[tte], tsc[tte])),
                    "tx_ap": float(average_precision_score(ty[tte], tsc[tte])),
                    "tx_f1": float(f1_score(ty[tte], tp, zero_division=0)), "tx_threshold": tthr,
                    "tx_base_rate": float(ty[tte].mean()), "tx_test_positives": int(ty[tte].sum())})
    return out


def _loss(logits, y, mask, pos_weight):
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask], pos_weight=pos_weight)


def tx_targets(prep: Prepared, v: GraphView):
    """Training transactions of a view: labels + positions of train positives / negatives."""
    y = prep.tx_y[v.tx_ids].astype(np.float32)
    tr = prep.tx_split[v.tx_ids] == 0
    return {"y": torch.as_tensor(y, device=prep.device),
            "pos": torch.as_tensor(np.flatnonzero(tr & (y == 1)), device=prep.device),
            "neg": torch.as_tensor(np.flatnonzero(tr & (y == 0)), device=prep.device)}


def _tx_loss(model, v, rep, tgt, gen, neg_ratio: int = 20, max_neg: int = 30000):
    """Transaction loss on all positives + a fresh random sample of negatives each step."""
    pos, neg = tgt["pos"], tgt["neg"]
    if len(pos) == 0 or len(neg) == 0:
        return rep.new_zeros(())
    k = min(len(neg), max(256, neg_ratio * len(pos)), max_neg)
    pick = torch.randint(len(neg), (k,), generator=gen).to(neg.device)
    sel = torch.cat([pos, neg[pick]])
    logit = model.tx_logits(v, rep, sel)
    pw = torch.tensor(min(POS_WEIGHT_CAP, k / len(pos)), device=rep.device)
    return F.binary_cross_entropy_with_logits(logit, tgt["y"][sel], pos_weight=pw)


def _val(prep, pred):
    how, ap = select_combine(prep, pred)
    va = prep.split == 1
    return ap, float(roc_auc_score(prep.y[va], combine(pred, how)[va])), how


# ------------------------------------------------------------------ centralised training
def train_central(kind: str, prep: Prepared, epochs: int = 120, lr: float = 3e-3, seed: int = 0,
                  progress=None) -> tuple:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = new_model(kind, prep).to(prep.device)
    v, s = prep.central, prep.central_sub
    own = v.acc_ids[: v.n_own]
    y = torch.as_tensor(prep.y[own], dtype=torch.float32, device=prep.device)
    tr = torch.as_tensor(prep.split[own] == 0, device=prep.device)
    pw = torch.tensor((1 - prep.y[own][prep.split[own] == 0]).sum() / max(prep.y[own][prep.split[own] == 0].sum(), 1),
                      device=prep.device).clamp(max=POS_WEIGHT_CAP)
    tgt = tx_targets(prep, v)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist, best, best_state = [], -1, None
    for ep in range(1, epochs + 1):
        model.train()
        logits, aux = model(v, s)
        loss = _loss(logits, y, tr, pw) + _tx_loss(model, v, aux["rep"], tgt, gen)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 5 == 0 or ep == epochs:
            vap, vauc, how = _val(prep, predict(model, prep, kind))
            hist.append({"step": ep, "loss": float(loss), "val_ap": vap, "val_auc": vauc, "combine": how})
            if vap > best:
                best, best_state = vap, copy.deepcopy(model.state_dict())
            if progress:
                progress(kind, ep / epochs, hist[-1])
    model.load_state_dict(best_state)
    return model, hist


# ------------------------------------------------------------------ federated training
def train_federated(prep: Prepared, rounds: int = 25, local_epochs: int = 4, lr: float = 3e-3,
                    exchange: bool = True, dp_noise: float = 0.0, clip_norm: float = 2.0,
                    emb_noise: float = 0.05, seed: int = 0, progress=None) -> tuple:
    """FedAvg over banks with secure aggregation, optional client-level DP and (optionally)
    the nested cross-bank embedding exchange."""
    kind = "nested_federated" if exchange else "nested_local"
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    gmodel = new_model("nested", prep).to(prep.device)
    gstate = copy.deepcopy(gmodel.state_dict())
    n_banks = len(prep.banks)
    bank_y, bank_tr, bank_pw, bank_tx = [], [], [], []
    for v in prep.banks:
        own = v.acc_ids[: v.n_own]
        bank_y.append(torch.as_tensor(prep.y[own], dtype=torch.float32, device=prep.device))
        trm = prep.split[own] == 0
        bank_tr.append(torch.as_tensor(trm, device=prep.device))
        bank_pw.append(torch.tensor(min(POS_WEIGHT_CAP, (1 - prep.y[own][trm]).sum() / max(prep.y[own][trm].sum(), 1)), device=prep.device))
        bank_tx.append(tx_targets(prep, v))
    hist, best, best_state = [], -1, None
    comm_bytes = 0
    for rnd in range(1, rounds + 1):
        gmodel.load_state_dict(gstate)
        ex = exchange_embeddings(gmodel, prep, emb_noise, seed=seed * 1000 + rnd) if exchange else (None, None)
        if exchange:
            comm_bytes += int(ex[1].sum().item()) * gmodel.hidden * 4
        deltas, losses, norms = [], [], []
        gvec = flat(gstate).cpu()
        for b, (v, s) in enumerate(zip(prep.banks, prep.bank_subs)):
            local = copy.deepcopy(gmodel)
            local.train()
            opt = torch.optim.AdamW(local.parameters(), lr=lr, weight_decay=1e-4)
            ge, gm = ghost_inputs(v, *ex)
            for _ in range(local_epochs):
                logits, aux = local(v, s, ge, gm)
                loss = _loss(logits, bank_y[b], bank_tr[b], bank_pw[b]) + _tx_loss(local, v, aux["rep"], bank_tx[b], gen)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                opt.step()
            d = flat(local.state_dict()).cpu() - gvec
            d, n = clip(d, clip_norm)
            deltas.append(d); losses.append(float(loss)); norms.append(n)
        total, _ = secure_aggregate(deltas, seed * 7919 + rnd)
        if dp_noise > 0:
            total = total + gaussian_noise(total.shape, dp_noise * clip_norm, seed * 104729 + rnd)
        comm_bytes += 2 * n_banks * gvec.numel() * 4
        gstate = unflat((gvec + total / n_banks).to(prep.device), gstate)
        gmodel.load_state_dict(gstate)
        ex_eval = exchange_embeddings(gmodel, prep, emb_noise, seed=seed * 1000 + rnd) if exchange else None
        vap, vauc, how = _val(prep, predict(gmodel, prep, kind, ex_eval))
        rec = {"step": rnd, "loss": float(np.mean(losses)), "bank_losses": losses, "update_norms": norms,
               "val_ap": vap, "val_auc": vauc, "combine": how, "comm_mb": comm_bytes / 1e6}
        hist.append(rec)
        if rec["val_ap"] > best:
            best, best_state = rec["val_ap"], copy.deepcopy(gstate)
        if progress:
            progress(kind, rnd / rounds, rec)
    gmodel.load_state_dict(best_state)
    privacy = {"dp_noise_multiplier": dp_noise, "clip_norm": clip_norm, "rounds": rounds,
               "epsilon_updates": rdp_epsilon(dp_noise, rounds) if dp_noise > 0 else None,
               "embedding_noise_std": emb_noise if exchange else None,
               "epsilon_embeddings": rdp_epsilon(emb_noise / 2.0, rounds) if exchange and emb_noise > 0 else None,
               "secure_aggregation": True, "communication_mb": comm_bytes / 1e6}
    return gmodel, hist, privacy


def train_all(prep: Prepared, cfg: dict | None = None, progress=None) -> dict:
    cfg = {"epochs": 120, "rounds": 25, "local_epochs": 4, "dp_noise": 0.0, "emb_noise": 0.05, **(cfg or {})}
    results, models = {}, {}
    for kind in ("mlp", "flat_gnn", "nested_central"):
        t = time.time()
        m, h = train_central(kind, prep, cfg["epochs"], progress=progress)
        results[kind] = {**evaluate(prep, predict(m, prep, kind)), "history": h, "train_s": time.time() - t, "label": MODEL_INFO[kind]}
        models[kind] = (m, None)
    for kind, exch in (("nested_local", False), ("nested_federated", True)):
        t = time.time()
        m, h, priv = train_federated(prep, cfg["rounds"], cfg["local_epochs"], exchange=exch,
                                     dp_noise=cfg["dp_noise"], emb_noise=cfg["emb_noise"], progress=progress)
        ex = exchange_embeddings(m, prep, cfg["emb_noise"]) if exch else None
        results[kind] = {**evaluate(prep, predict(m, prep, kind, ex)), "history": h, "privacy": priv, "train_s": time.time() - t,
                         "label": MODEL_INFO[kind]}
        models[kind] = (m, ex)
    return {"results": results, "models": models, "config": cfg}
