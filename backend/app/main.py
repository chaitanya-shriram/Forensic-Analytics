"""FastAPI service for the cross-bank fraud ring detection console."""
from __future__ import annotations

import asyncio
import math
import mimetypes
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import DAY, TX_MERCHANT, TX_TRANSFER
from .engine import ACTIVE, Engine
from .explain import explain_account
from .features import FEATURE_LABELS, feature_names
from .privacy import pseudonym, rdp_epsilon
from .train import MODEL_INFO


def clean(o):
    """Make engine output JSON safe (numpy scalars, sets, NaN/inf)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Hub:
    """Fan-out of engine events (from worker threads) to websocket clients."""

    def __init__(self):
        self.queues: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, ev):
        if self.loop is None:
            return
        ev = clean(ev)
        for q in list(self.queues):
            self.loop.call_soon_threadsafe(self._put, q, ev)

    @staticmethod
    def _put(q, ev):
        if q.qsize() > 500:
            return
        q.put_nowait(ev)


hub = Hub()
engine = Engine(seed=int(os.environ.get("FRAUD_SEED", 7)),
                device=os.environ.get("FRAUD_DEVICE") or None,
                dataset=os.environ.get("FRAUD_DATASET", "ibm_aml"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    engine.listeners.append(hub.publish)
    if os.environ.get("FRAUD_NO_AUTOSTART") != "1":
        threading.Thread(target=_boot, daemon=True).start()
    yield
    engine.stop_stream()


def _boot():
    try:
        engine.start(force_train=os.environ.get("FRAUD_RETRAIN") == "1")
    except Exception as e:  # surfaced through /api/status
        engine.status.update(state="error", message=repr(e))
        raise


app = FastAPI(title="Cross-Bank Fraud Ring Detection", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def need_ready():
    if not engine.ready:
        raise HTTPException(503, detail=engine.status)


# ---------------------------------------------------------------------------- status / overview
@app.get("/api/status")
def status():
    return clean({**engine.status, "ready": engine.ready, "device": engine.device})


@app.get("/api/overview")
def overview():
    need_ready()
    w, s, thr = engine.world, engine.scores, engine.threshold
    with engine.lock:
        n = w.tx.n
        kind, src, dst, amt, ts = (w.tx.view(c).copy() for c in ("kind", "src", "dst", "amount", "ts"))
        banks = []
        for b in range(w.n_banks):
            own = w.acc_bank == b
            v = engine.prep.banks[b]
            banks.append({"id": b, "name": w.bank_names[b], "accounts": int(own.sum()), "flagged": int((s[own] >= thr).sum()),
                          "mean_score": float(s[own].mean()), "external_counterparties": int(v.n_ghost),
                          "devices": int(len(v.dev_ids)), "merchants": int(len(v.mer_ids))})
        tr = kind == TX_TRANSFER
        cross = tr & (w.acc_bank[src] != w.acc_bank[np.maximum(dst, 0)])
        day = (ts // DAY).astype(int)
        risky = np.maximum(s[src], np.where(dst >= 0, s[np.maximum(dst, 0)], 0)) >= thr
        days = []
        for d in range(int(day.min()), int(day.max()) + 1):
            m = day == d
            days.append({"day": engine.fmt_time(d * DAY)[:10], "tx": int(m.sum()), "volume": float(amt[m].sum()),
                         "suspicious_volume": float(amt[m & risky].sum()), "suspicious_tx": int((m & risky).sum())})
        hist, edges = np.histogram(s[w.is_member], bins=20, range=(0, 1))
        patterns = {}
        for r in engine.rings:
            patterns[r["pattern_label"]] = patterns.get(r["pattern_label"], 0) + 1
        res = engine.results[ACTIVE]
        return clean({
            "accounts": int(w.is_member.sum()), "external_accounts": int((~w.is_member).sum()), "banks": banks,
            "dataset": w.source, "transactions": n, "transfers": int(tr.sum()),
            "cross_bank_transfers": int(cross.sum()), "devices": w.n_devices, "merchants": w.n_merchants,
            "flagged": int((s >= thr).sum()), "threshold": thr, "rings": len(engine.rings),
            "cross_bank_rings": sum(1 for r in engine.rings if len(r["banks"]) > 1),
            "suspicious_flow": float(sum(r["total_flow"] for r in engine.rings)),
            "alerts": len(engine.alerts), "score_hist": [{"bin": f"{edges[i]:.2f}", "count": int(hist[i])} for i in range(20)],
            "patterns": [{"pattern": k, "count": v} for k, v in patterns.items()], "daily": days,
            "thresholds": thresholds(),
            "active_model": {"key": ACTIVE, "label": MODEL_INFO[ACTIVE], **{k: res[k] for k in ("auc", "ap", "f1", "precision", "recall", "ring_recall")}},
            "stream": engine.stream_stats(),
        })


@app.get("/api/dataset")
def dataset():
    need_ready()
    w = engine.world
    info = {"source": w.source, "n_banks": w.n_banks, "bank_names": w.bank_names[: w.n_banks],
            "accounts": int(w.is_member.sum()), "external_accounts": int((~w.is_member).sum()),
            "history_tx": int(w.tx.n), "future_tx": len(w.future), "rings": len(w.rings),
            "positives": int(w.acc_label[w.is_member].sum()), "features": feature_names(w)}
    if w.source == "ibm_aml":
        info.update({"report": w.meta, "citation": "Altman et al., Realistic Synthetic Financial Transactions for "
                     "Anti-Money Laundering Models, NeurIPS 2023 Datasets & Benchmarks (IBM AMLworld HI-Small)",
                     "patterns": {p: sum(r["pattern"] == p for r in w.rings) for p in sorted({r["pattern"] for r in w.rings})}})
    return clean(info)


@app.get("/api/flows")
def flows():
    """Money flow between consortium members (and external institutions) for the 3D consortium view."""
    need_ready()
    w, s, thr = engine.world, engine.scores, engine.threshold
    with engine.lock:
        src, dst, amt = (w.tx.view(c).copy() for c in ("src", "dst", "amount"))
    m = dst >= 0
    src, dst, amt = src[m], dst[m], amt[m]
    K = w.n_banks
    bs, bd = w.acc_bank[src], w.acc_bank[dst]
    hot = (s[src] >= thr) | (s[dst] >= thr)
    key = bs * (K + 1) + bd
    tot = np.bincount(key, weights=amt, minlength=(K + 1) ** 2)
    cnt = np.bincount(key, minlength=(K + 1) ** 2)
    sus = np.bincount(key, weights=amt * hot, minlength=(K + 1) ** 2)
    links = []
    for i in range(K + 1):
        for j in range(K + 1):
            k = i * (K + 1) + j
            if i != j and cnt[k]:
                links.append({"source": i, "target": j, "volume": float(tot[k]), "count": int(cnt[k]),
                              "suspicious": float(sus[k])})
    nodes = []
    for b in range(K + 1):
        own = w.acc_bank == b
        nodes.append({"id": b, "name": w.bank_name(b), "external": b == K, "accounts": int(own.sum()),
                      "flagged": int((s[own] >= thr).sum()) if b < K else 0,
                      "internal_volume": float(tot[b * (K + 1) + b])})
    return clean({"nodes": nodes, "links": links})


@app.get("/api/metrics")
def metrics():
    need_ready()
    out = []
    for k, r in engine.results.items():
        out.append({"key": k, **{kk: vv for kk, vv in r.items()}})
    n_params = sum(p.numel() for p in engine.model.parameters())
    return clean({"models": out, "active": ACTIVE, "n_params": n_params, "experiments": engine.experiments,
                  "split": {"train": int((engine.prep.split == 0).sum()), "val": int((engine.prep.split == 1).sum()),
                            "test": int((engine.prep.split == 2).sum())},
                  "timings": engine.prep.timings})


@app.get("/api/privacy")
def privacy():
    need_ready()
    p = engine.results[ACTIVE].get("privacy", {})
    shared = [
        {"what": "Model parameter updates", "how": "Clipped (L2 <= %.1f) and pairwise-masked (secure aggregation); server only sees the sum" % p.get("clip_norm", 2.0)},
        {"what": "Cross-bank account embeddings", "how": "64-d subgraph embedding, norm-clipped to 1 and Gaussian-noised (std %.2f), keyed by HMAC pseudonym" % (p.get("embedding_noise_std") or 0)},
        {"what": "Device sharing counts", "how": "PSI-style registry over HMAC(device id); banks learn only counts of accounts / banks per device"},
        {"what": "Flagged accounts for ring assembly", "how": "Only pseudonyms of accounts above threshold and their pseudonymised links"},
    ]
    never = ["Raw transactions", "Customer identities / KYC data", "Account balances", "Unflagged customers' features"]
    return clean({**p, "shared": shared, "never_shared": never,
                  "note": "epsilon values use a conservative RDP bound for the Gaussian mechanism without subsampling amplification."})


# ---------------------------------------------------------------------------- accounts
@app.get("/api/accounts")
def accounts(bank: int | None = None, q: str | None = None, flagged: bool | None = None, limit: int = 50,
             offset: int = 0, sort: str = "score"):
    need_ready()
    w = engine.world
    idx = np.flatnonzero(w.is_member)
    if bank is not None:
        idx = idx[w.acc_bank[idx] == bank]
    if flagged is not None:
        idx = idx[(engine.scores[idx] >= engine.threshold) == flagged]
    if q:
        ql = q.lower()
        idx = np.array([a for a in idx if ql in w.account_name(a).lower() or ql in pseudonym("acc", a)], dtype=int)
    if sort == "score":
        idx = idx[np.argsort(-engine.scores[idx], kind="stable")]
    return clean({"total": len(idx), "items": [engine.account_summary(int(a)) for a in idx[offset:offset + limit]]})


def account_tx(a: int, limit: int = 40):
    t = engine.world.tx
    m = np.flatnonzero((t.view("src") == a) | (t.view("dst") == a))[-limit:][::-1]
    return [engine._tx_record(int(i)) for i in m]


def ego_graph(a: int, max_nodes: int = 60):
    w, s = engine.world, engine.scores
    t = w.tx
    tr = t.view("kind") == TX_TRANSFER
    src, dst, amt = t.view("src")[tr], t.view("dst")[tr], t.view("amount")[tr]
    nodes, edges = {}, {}

    def add_acc(x, hop):
        if x not in nodes and len(nodes) < max_nodes:
            nodes[x] = {"id": f"a{x}", "kind": "account", "account": int(x), "label": w.account_name(x),
                        "bank": int(w.acc_bank[x]), "score": float(s[x]), "flagged": bool(s[x] >= engine.threshold),
                        "hop": hop, "truth": int(w.acc_label[x])}
        return x in nodes

    def add_edge(u, v, amount):
        k = (u, v)
        e = edges.setdefault(k, {"source": u, "target": v, "amount": 0.0, "count": 0, "kind": "transfer"})
        e["amount"] += float(amount); e["count"] += 1

    add_acc(a, 0)
    m = (src == a) | (dst == a)
    pairs = {}
    for x, y, v in zip(src[m], dst[m], amt[m]):
        pairs[(int(x), int(y))] = pairs.get((int(x), int(y)), 0) + v
    for (x, y), v in sorted(pairs.items(), key=lambda kv: -kv[1])[:30]:
        o = y if x == a else x
        if add_acc(o, 1):
            add_edge(f"a{x}", f"a{y}", v)
    hop1 = [x for x, n in nodes.items() if n["hop"] == 1 and n["flagged"]]
    for h in hop1:  # follow money through flagged neighbours
        mm = (src == h) | (dst == h)
        for x, y, v in zip(src[mm], dst[mm], amt[mm]):
            o = int(y) if x == h else int(x)
            if s[o] >= engine.threshold and add_acc(o, 2):
                add_edge(f"a{int(x)}", f"a{int(y)}", v)
    dev_users = w.device_accounts()
    for d in w.acc_devices[a]:
        users = dev_users.get(d, [])
        if len(users) > 1:
            did = f"d{d}"
            nodes[did] = {"id": did, "kind": "device", "label": f"dev {pseudonym('dev', d)[:6]}", "shared": len(users)}
            for u in users[:15]:
                if add_acc(u, 1):
                    edges[(f"a{u}", did)] = {"source": f"a{u}", "target": did, "kind": "device", "amount": 0, "count": 0}
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


@app.get("/api/accounts/{a}")
def account(a: int):
    need_ready()
    if not 0 <= a < engine.world.n_accounts:
        raise HTTPException(404, "unknown account")
    with engine.lock:
        exp = explain_account(engine, a) if engine.world.acc_bank[a] < engine.world.n_banks else None
        return clean({**engine.account_summary(a), "explanation": exp,
                      "model_scores": {k: float(v[a]) for k, v in engine.model_scores.items()},
                      "features_raw": {FEATURE_LABELS[f]: float(engine.X_raw[a, i]) for i, f in enumerate(feature_names(engine.world))},
                      "transactions": account_tx(a), "graph": ego_graph(a),
                      "rings": [r["id"] for r in engine.rings if a in r["members"]]})


def thresholds():
    return {"tx": engine.tx_threshold, "account": engine.threshold,
            "ring": getattr(engine, "ring_thr", None), "combine": engine.combine_how}


@app.get("/api/review")
def review(limit: int = 40, flagged_only: bool = True):
    """Investigator queue: riskiest transactions first, with both parties' account risk and reasons."""
    need_ready()
    from .explain import reasons
    w = engine.world
    with engine.lock:
        n = w.tx.n
        r = np.nan_to_num(engine.tx_risk[:n], nan=-1.0)
        idx = np.argsort(-r)[: max(limit * 4, limit)]
        if flagged_only:
            idx = idx[r[idx] >= engine.tx_threshold]
        items = []
        for i in idx[:limit]:
            rec = engine._tx_record(int(i))
            for side in ("src", "dst"):
                a = rec[side]
                if a is not None and a >= 0 and w.acc_bank[a] < w.n_banks:
                    rec[f"{side}_score"] = float(engine.scores[a])
                    rec[f"{side}_reasons"] = reasons(w, engine.X_raw, a)[:2]
            items.append(rec)
        total_flagged = int((r >= engine.tx_threshold).sum())
    return clean({"items": items, "total_flagged": total_flagged, "thresholds": thresholds()})


@app.get("/api/nested/sample")
def nested_sample():
    """A good account to demonstrate the nested GNN: a caught launderer from the test split."""
    need_ready()
    w, s, sp = engine.world, engine.scores, engine.prep.split
    for key in ("8001555A0",):
        for a in np.flatnonzero(w.is_member):
            if w.account_name(a).endswith(key):
                return {"account": int(a)}
    cand = np.flatnonzero((sp == 2) & (w.acc_label == 1) & (s >= engine.threshold))
    if not len(cand):
        cand = np.flatnonzero(w.is_member)
    return {"account": int(cand[np.argmax(s[cand])])}


@app.get("/api/nested/{a}")
def nested(a: int):
    """Everything needed to walk one account through the nested GNN, step by step."""
    need_ready()
    w = engine.world
    if not 0 <= a < w.n_accounts or w.acc_bank[a] >= w.n_banks:
        raise HTTPException(404, "pick an account held by a member bank")
    with engine.lock:
        exp = explain_account(engine, a)
        names = feature_names(w)
        Xm = engine.X_raw[w.is_member]
        feats = []
        for f in exp["features"][:8]:
            k = names.index(f["feature"])
            feats.append({**f, "percentile": float((Xm[:, k] < engine.X_raw[a, k]).mean() * 100)})
        t = w.tx
        mine = np.flatnonzero((t.view("src") == a) | (t.view("dst") == a))
        r = np.nan_to_num(engine.tx_risk[: t.n][mine], nan=0.0)
        top = mine[np.argsort(-r)][:8]
        txs = [engine._tx_record(int(i)) for i in top]
        return clean({**engine.account_summary(a), "explanation": {**exp, "features": feats},
                      "transactions": txs, "n_transactions": int(len(mine)),
                      "thresholds": thresholds(), "rings": [x["id"] for x in engine.rings if a in x["members"]]})


# ---------------------------------------------------------------------------- rings
@app.get("/api/rings")
def rings():
    need_ready()
    return clean([engine.ring_summary(r) for r in engine.rings])


@app.get("/api/rings/{rid}")
def ring(rid: int):
    need_ready()
    r = next((r for r in engine.rings if r["id"] == rid), None)
    if r is None:
        raise HTTPException(404, "unknown ring")
    w, s = engine.world, engine.scores
    mem = set(r["members"])
    t = w.tx
    sel = np.flatnonzero((t.view("kind") == TX_TRANSFER) & (np.isin(t.view("src"), r["members"]) | np.isin(t.view("dst"), r["members"])))
    nodes, edges, inflow = {}, {}, {}
    for a in r["members"]:
        nodes[f"a{a}"] = {"id": f"a{a}", "kind": "account", "account": a, "label": w.account_name(a), "bank": int(w.acc_bank[a]),
                          "score": float(s[a]), "member": True, "truth": int(w.acc_label[a])}
    timeline = []
    for i in sel:
        x, y, v = int(t.src[i]), int(t.dst[i]), float(t.amount[i])
        if x in mem and y in mem:
            e = edges.setdefault((x, y), {"source": f"a{x}", "target": f"a{y}", "amount": 0.0, "count": 0, "kind": "transfer"})
            e["amount"] += v; e["count"] += 1
            timeline.append(engine._tx_record(int(i)))
        elif y in mem:
            inflow[(x, y)] = inflow.get((x, y), 0) + v
    for (x, y), v in sorted(inflow.items(), key=lambda kv: -kv[1])[:12]:
        nodes.setdefault(f"a{x}", {"id": f"a{x}", "kind": "account", "account": x, "label": w.account_name(x),
                                   "bank": int(w.acc_bank[x]), "score": float(s[x]), "member": False, "truth": int(w.acc_label[x])})
        edges[(x, y)] = {"source": f"a{x}", "target": f"a{y}", "amount": v, "count": 1, "kind": "inflow"}
    dev = {}
    for a in r["members"]:
        for d in w.acc_devices[a]:
            dev.setdefault(d, []).append(a)
    for d, us in dev.items():
        if len(us) > 1:
            nodes[f"d{d}"] = {"id": f"d{d}", "kind": "device", "label": f"dev {pseudonym('dev', d)[:6]}", "shared": len(us)}
            for u in us:
                edges[(u, f"d{d}")] = {"source": f"a{u}", "target": f"d{d}", "kind": "device", "amount": 0, "count": 0}
    timeline.sort(key=lambda x: x["ts"])
    members = [engine.account_summary(a) for a in r["members"]]
    truth = None
    if r["matched_true_ring"] >= 0:
        tr_ = w.rings[r["matched_true_ring"]]
        tm = set(tr_.get("all_members", tr_["members"]))
        truth = {"id": tr_["id"], "pattern": tr_["pattern"], "size": len(tm),
                 "recall": len(mem & tm) / len(tm), "live": bool(tr_.get("live"))}
    return clean({**engine.ring_summary(r), "members": members, "graph": {"nodes": list(nodes.values()), "edges": list(edges.values())},
                  "timeline": timeline[-80:], "truth": truth})


# ---------------------------------------------------------------------------- live stream
@app.get("/api/alerts")
def alerts(limit: int = 100):
    need_ready()
    return clean(list(engine.alerts)[:limit])


@app.get("/api/transactions/recent")
def recent(limit: int = 100):
    need_ready()
    return clean(list(engine.recent_tx)[:limit])


class TxIn(BaseModel):
    src: int
    dst: int | None = None
    merchant: int | None = None
    amount: float = Field(gt=0)


@app.post("/api/transactions")
def submit_tx(tx: TxIn):
    need_ready()
    w = engine.world
    if not 0 <= tx.src < w.n_accounts or (tx.dst is not None and not 0 <= tx.dst < w.n_accounts):
        raise HTTPException(400, "unknown account")
    if (tx.dst is None) == (tx.merchant is None):
        raise HTTPException(400, "give exactly one of dst or merchant")
    if tx.merchant is not None and not 0 <= tx.merchant < w.n_merchants:
        raise HTTPException(400, "unknown merchant")
    engine.sim_time += 60
    row = (tx.src, tx.dst if tx.dst is not None else -1, tx.merchant if tx.merchant is not None else -1, tx.amount,
           engine.sim_time, TX_TRANSFER if tx.dst is not None else TX_MERCHANT)
    before = {a: float(engine.scores[a]) for a in (tx.src, tx.dst) if a is not None}
    out = engine.process([row])
    hub.publish({"type": "batch", "sim_time": engine.sim_time, "time": engine.fmt_time(engine.sim_time), **out,
                 "stats": engine.stream_stats()})
    return clean({**out, "before": before, "after": {a: float(engine.scores[a]) for a in before}})


class StreamCfg(BaseModel):
    speed: float | None = None
    rate: float | None = None
    ring_every: int | None = None


@app.post("/api/stream/start")
def stream_start(cfg: StreamCfg | None = None):
    need_ready()
    engine.start_stream(**(cfg.model_dump() if cfg else {}))
    return clean(engine.stream_stats())


@app.post("/api/stream/stop")
def stream_stop():
    engine.stop_stream()
    return clean(engine.stream_stats())


@app.get("/api/stream")
def stream_state():
    need_ready()
    return clean({**engine.stream_stats(), "latencies": list(engine.latencies)[-60:]})


@app.post("/api/stream/inject-ring")
def inject():
    need_ready()
    r = engine.inject_ring()
    if r is None:
        raise HTTPException(409, "no sleeper accounts left to form a new ring")
    return clean({"id": r["id"], "pattern": r["pattern"], "banks": r["banks"], "size": len(r["members"]),
                  "pending_tx": len(engine.pending)})


# ---------------------------------------------------------------------------- experiments
class ExperimentIn(BaseModel):
    rounds: int = Field(15, ge=1, le=60)
    local_epochs: int = Field(4, ge=1, le=10)
    dp_noise: float = Field(0.0, ge=0, le=5)
    emb_noise: float = Field(0.05, ge=0, le=2)
    exchange: bool = True


@app.post("/api/experiments")
def run_experiment(cfg: ExperimentIn):
    need_ready()
    if engine.status.get("state") == "training":
        raise HTTPException(409, "training already running")
    engine.status.update(state="training", progress=0.0, message="Queued federated experiment")
    threading.Thread(target=engine.retrain_federated, kwargs=cfg.model_dump(), daemon=True).start()
    eps = rdp_epsilon(cfg.dp_noise, cfg.rounds) if cfg.dp_noise > 0 else None
    return clean({"started": True, "epsilon_estimate": eps})


@app.get("/api/experiments")
def experiments():
    return clean([{k: v for k, v in e.items()} for e in engine.experiments])


# ---------------------------------------------------------------------------- websocket
@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q: asyncio.Queue = asyncio.Queue()
    hub.queues.add(q)
    try:
        await sock.send_json(clean({"type": "status", **engine.status, "ready": engine.ready}))
        while True:
            ev = await q.get()
            await sock.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.queues.discard(q)


# ---------------------------------------------------------------------------- frontend (built)
# Windows can map .js to text/plain via the registry, which browsers reject for module scripts
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
