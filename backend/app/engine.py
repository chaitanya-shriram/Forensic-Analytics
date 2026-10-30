"""Stateful detection engine: model lifecycle, incremental near-real-time scoring, alerts,
ring tracking and the transaction stream simulator."""
from __future__ import annotations

import collections
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .data import DAY, TX_MERCHANT, TX_TRANSFER, activate_live_ring, generate_world, normal_traffic
from .features import account_features, device_features, device_stats, tx_frame
from .graph import build_view, extract_chunks, subs_to
from .privacy import gaussian_noise, pseudonym, psi_device_counts
from .rings import detect_rings, detect_tx_rings, tx_ring_threshold
from .train import (MODEL_INFO, account_tx_max, combine, evaluate, exchange_embeddings, ghost_inputs, new_model, predict, prepare,
                    train_all, train_federated)

ARTIFACTS = Path(os.environ.get("FRAUD_ARTIFACTS", Path(__file__).resolve().parent.parent / "artifacts"))
ACTIVE = "nested_federated"
EVAL_VERSION = 2
SPLITS = {0: "train", 1: "val", 2: "test", -1: "external", 3: "live"}


class Engine:
    def __init__(self, seed: int = 7, device: str | None = None, world_kwargs: dict | None = None,
                 train_cfg: dict | None = None, artifacts: Path | None = None, dataset: str = "synthetic"):
        self.seed = seed
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.world_kwargs = world_kwargs or {}
        self.train_cfg = train_cfg or ({"epochs": 200, "rounds": 40, "local_epochs": 4} if dataset == "ibm_aml" else {})
        self.artifacts = Path(artifacts or ARTIFACTS)
        self.lock = threading.RLock()
        self.status = {"state": "starting", "progress": 0.0, "message": "Loading consortium data"}
        self.replay_pos = 0
        self.listeners: list = []  # callables(event dict)
        self.alerts = collections.deque(maxlen=300)
        self.recent_tx = collections.deque(maxlen=200)
        self.latencies = collections.deque(maxlen=200)
        self.experiments: list = []
        self.stream_thread = None
        self.stream_on = False
        self.stream_cfg = {"speed": 120.0 if dataset == "ibm_aml" else 600.0, "rate": 6.0, "ring_every": 25}
        self.ticks = 0
        self.live_rings: dict = {}
        self.pending: list = []
        self._flag_key = None
        self.ready = False

    # ------------------------------------------------------------------ lifecycle
    def emit(self, ev: dict):
        for f in list(self.listeners):
            try:
                f(ev)
            except Exception:
                pass

    def _progress(self, kind, frac, rec):
        order = ["mlp", "flat_gnn", "nested_central", "nested_local", "nested_federated"]
        i = order.index(kind) if kind in order else 0
        self.status.update(state="training", progress=(i + frac) / len(order),
                           message=f"Training {MODEL_INFO.get(kind, kind)} ({frac:.0%})")
        self.emit({"type": "training", "model": kind, "progress": frac, "record": rec})

    def start(self, force_train: bool = False):
        if self.dataset == "ibm_aml":
            from .aml import load_ibm_aml
            self.status.update(state="loading", message="Loading IBM AML transactions (first run downloads + cleans ~5M rows)")
            self.world = load_ibm_aml(**self.world_kwargs)
        else:
            self.world = generate_world(seed=self.seed, **self.world_kwargs)
        self.status.update(state="preparing", message="Building bank graph views and nested subgraphs")
        self.prep = prepare(self.world, seed=0, device=self.device)
        self.X_raw = self.prep.X_raw
        tag = f"{self.dataset}_seed{self.seed}_" + "_".join(f"{k}{v}" for k, v in sorted(self.world_kwargs.items()))
        mdir = self.artifacts / tag
        if not force_train and (mdir / "results.json").exists():
            self.status.update(state="loading", message="Loading trained models")
            saved = json.loads((mdir / "results.json").read_text())
            self.results = saved["results"]
            self.models = {}
            for kind in MODEL_INFO:
                m = new_model("nested" if kind.startswith("nested") else kind, self.prep).to(self.device)
                m.load_state_dict(torch.load(mdir / f"{kind}.pt", map_location=self.device))
                self.models[kind] = m
            if saved.get("eval_version") != EVAL_VERSION:  # metrics definition changed: re-score cached models
                self.status.update(message="Re-evaluating cached models")
                for kind, m in self.models.items():
                    ex = exchange_embeddings(m, self.prep, 0.05) if kind == "nested_federated" else None
                    self.results[kind].update(evaluate(self.prep, predict(m, self.prep, kind, ex)))
                saved.update(results=self.results, eval_version=EVAL_VERSION)
                (mdir / "results.json").write_text(json.dumps(saved))
        else:
            out = train_all(self.prep, self.train_cfg, progress=self._progress)
            self.results = out["results"]
            self.models = {k: m for k, (m, _) in out["models"].items()}
            mdir.mkdir(parents=True, exist_ok=True)
            for k, m in self.models.items():
                torch.save(m.state_dict(), mdir / f"{k}.pt")
            (mdir / "results.json").write_text(json.dumps({"results": self.results, "config": out["config"],
                                                          "eval_version": EVAL_VERSION}))
        self.model = self.models[ACTIVE]
        self.emb_noise = self.results[ACTIVE].get("privacy", {}).get("embedding_noise_std") or 0.05
        self.threshold = self.results[ACTIVE]["threshold"]
        self.combine_how = self.results[ACTIVE].get("combine", "acc")
        self.tx_threshold = self.results[ACTIVE].get("tx_threshold", 0.5)
        self.live_tx = {"seen": 0, "laundering": 0, "flagged": 0, "flagged_laundering": 0}
        self._full_refresh()
        self.sim_time = float(self.world.tx.view("ts").max())
        self.ready = True
        self.status.update(state="ready", progress=1.0, message="Ready")
        self.emit({"type": "status", **self.status})

    # ------------------------------------------------------------------ scoring
    def ghost_inputs(self, b: int):
        return ghost_inputs(self.prep.banks[b], self.pub, self.has)

    @torch.no_grad()
    def _full_refresh(self):
        """Full federated inference: exchange embeddings, encode every bank, cache z."""
        m = self.model
        m.eval()
        self.prep.bank_subs = [subs_to(extract_chunks(v), self.device) for v in self.prep.banks]
        self.pub, self.has = exchange_embeddings(m, self.prep, self.emb_noise, seed=self.seed)
        self.acc_prob = np.zeros(self.world.n_accounts)
        self.tx_risk = np.full(max(self.world.tx.n * 2, 1024), np.nan)
        self.z_cache = []
        for b, (v, s) in enumerate(zip(self.prep.banks, self.prep.bank_subs)):
            ge, gm = self.ghost_inputs(b)
            z, _ = m.encode_subgraphs(v, s, ge, gm)
            logits, h, _ = m.outer_forward(v, z, ge, gm)
            self.z_cache.append(z)
            self.acc_prob[v.acc_ids[: v.n_own]] = torch.sigmoid(logits).cpu().numpy()
            self._score_tx(v, m.representation(v, h, z), v.tx_ids, np.arange(len(v.tx_ids)))
        self.tx_max = account_tx_max(self.world, self.tx_risk[: self.world.tx.n])
        self._combine()
        self.model_scores = {ACTIVE: self.scores.copy()}
        for kind, mm in self.models.items():
            if kind != ACTIVE:
                ex = (self.pub, self.has) if kind == "nested_federated" else None
                self.model_scores[kind] = combine(predict(mm, self.prep, kind, ex), self.results[kind].get("combine", "acc"))
        self.rings = self._assemble_rings()

    def _assemble_rings(self):
        if self.world.source == "ibm_aml":
            n = self.world.tx.n
            if getattr(self, "ring_thr", None) is None:
                self.ring_thr = tx_ring_threshold(self.tx_risk[:n], self.tx_threshold)
            return detect_tx_rings(self.world, self.tx_risk[:n], self.ring_thr)
        return detect_rings(self.world, self.scores, self.threshold)

    def _combine(self):
        self.scores = combine({"acc": self.acc_prob, "txmax": self.tx_max}, self.combine_how)

    def _score_tx(self, v, rep, ids, pos):
        """Transaction risk for view-local transaction positions `pos` (global ids `ids`)."""
        if len(pos) == 0:
            return
        if len(self.tx_risk) < self.world.tx.n:
            self.tx_risk = np.concatenate([self.tx_risk, np.full(len(self.tx_risk), np.nan)])
        p = torch.sigmoid(self.model.tx_logits(v, rep, torch.as_tensor(pos, device=self.device))).cpu().numpy()
        self.tx_risk[ids] = np.fmax(self.tx_risk[ids], p)

    def _publish(self, b: int, own_pos: np.ndarray, z: torch.Tensor):
        """Re-publish (clipped + noised) embeddings of updated accounts for other banks."""
        v = self.prep.banks[b]
        z = z * torch.clamp(1.0 / (z.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        z = z + gaussian_noise(z.shape, self.emb_noise, int(self.ticks * 97 + b)).to(z.device)
        ids = torch.as_tensor(v.acc_ids[own_pos], device=self.device)
        self.pub[ids] = z
        self.has[ids] = True

    @torch.no_grad()
    def process(self, rows: list) -> dict:
        """Ingest a micro-batch of transactions and incrementally rescore affected accounts."""
        t0 = time.perf_counter()
        w = self.world
        with self.lock:
            first_new = w.tx.n
            w.tx.extend(rows)
            touched = sorted({int(r[0]) for r in rows} | {int(r[1]) for r in rows if r[1] >= 0})
            if not touched:
                return {}
            old_nd = len(self.prep.X_dev)
            if w.n_devices and w.n_devices > old_nd:  # a ring activation created new devices: refresh PSI counts + device features
                ds = device_stats(w)
                n_acc, n_banks = psi_device_counts(w.acc_devices, w.acc_bank, w.n_devices)
                self.prep.dev_stats = (n_acc, n_banks, ds[2])
                self.prep.X_dev = device_features(w, self.prep.dev_stats)
                touched = sorted(set(touched) | {a for a in range(w.n_accounts) if any(d >= old_nd for d in w.acc_devices[a])})
            Xr = account_features(w, touched, dev_stats=self.prep.dev_stats)
            self.X_raw[touched] = Xr
            self.prep.X[touched] = self.prep.norm(Xr)
            t_feat = time.perf_counter()

            touched_arr = np.array(touched)
            df = tx_frame(w)
            old = self.scores.copy()
            banks = sorted({int(w.acc_bank[a]) for a in touched if w.acc_bank[a] < w.n_banks})
            for b, v in enumerate(self.prep.banks):  # banks that see a touched account as a ghost
                if b not in banks and np.isin(v.acc_ids[v.n_own:], touched_arr).any():
                    banks.append(b)
            n_rescored = 0
            m = self.model
            for b in banks:
                v = build_view(w, self.prep.X, self.prep.X_dev, b, df).to(self.device)
                self.prep.banks[b] = v
                # dirty roots: own accounts within 2 hops of a touched node
                ptr, cdst = v.csr_ptr, v.csr_dst
                mer_start = v.n_acc + len(v.dev_ids)
                pos = np.full(w.n_accounts, -1)
                pos[v.acc_ids] = np.arange(v.n_acc)
                frontier = set(pos[touched_arr][pos[touched_arr] >= 0].tolist())
                seen = set(frontier)
                for _ in range(2):
                    nxt = set()
                    for u in frontier:
                        if u >= mer_start:  # merchants are never expanded inside subgraphs
                            continue
                        nxt.update(int(x) for x in cdst[ptr[u]:ptr[u + 1]] if x < mer_start)
                    frontier = nxt - seen
                    seen |= nxt
                dirty = np.array(sorted(i for i in seen if i < v.n_own), dtype=np.int64)
                ge, gm = self.ghost_inputs(b)
                if len(dirty):
                    sub = subs_to(extract_chunks(v, dirty), self.device)
                    z, _ = m.encode_subgraphs(v, sub, ge, gm)
                    self.z_cache[b][torch.as_tensor(dirty, device=self.device)] = z
                    tpos = dirty[np.isin(v.acc_ids[dirty], touched_arr)]
                    if len(tpos):
                        self._publish(b, tpos, self.z_cache[b][torch.as_tensor(tpos, device=self.device)])
                    n_rescored += len(dirty)
                logits, h, _ = m.outer_forward(v, self.z_cache[b], ge, gm)
                self.acc_prob[v.acc_ids[: v.n_own]] = torch.sigmoid(logits).cpu().numpy()
                new_ids = np.arange(first_new, w.tx.n)
                vpos = np.searchsorted(v.tx_ids, new_ids)
                ok = (vpos < len(v.tx_ids)) & (v.tx_ids[np.minimum(vpos, len(v.tx_ids) - 1)] == new_ids)
                self._score_tx(v, m.representation(v, h, self.z_cache[b]), new_ids[ok], vpos[ok])
            # newest transactions raise their endpoints' transaction-risk maxima
            new_ids = np.arange(first_new, w.tx.n)
            r = np.nan_to_num(self.tx_risk[new_ids])
            np.maximum.at(self.tx_max, w.tx.src[new_ids], r)
            dd = w.tx.dst[new_ids]
            np.maximum.at(self.tx_max, dd[dd >= 0], r[dd >= 0])
            self.tx_max[~w.is_member] = 0
            self._combine()
            self.model_scores[ACTIVE] = self.scores
            t_score = time.perf_counter()

            thr = self.threshold
            crossed = np.flatnonzero((old < thr) & (self.scores >= thr) & w.is_member)
            new_alerts = []
            for a in crossed:
                al = self._alert(int(a), self.sim_time)
                self.alerts.appendleft(al)
                new_alerts.append(al)
            r = np.nan_to_num(self.tx_risk[new_ids])
            hot = r >= self.tx_threshold
            lab = w.tx.label[new_ids] == 1
            self.live_tx["seen"] += len(new_ids)
            self.live_tx["laundering"] += int(lab.sum())
            self.live_tx["flagged"] += int(hot.sum())
            self.live_tx["flagged_laundering"] += int((hot & lab).sum())
            new_rings = []
            flag_key = hash(np.flatnonzero(self.scores >= thr).tobytes()) ^ hash(int((np.nan_to_num(self.tx_risk[: w.tx.n]) >= (getattr(self, "ring_thr", None) or 1.0)).sum()))
            if flag_key != self._flag_key or self.ticks % 10 == 0:  # re-assemble rings only when flags change
                self._flag_key = flag_key
                prev_rings = {tuple(r["members"]) for r in self.rings}
                self.rings = self._assemble_rings()
                new_rings = [r for r in self.rings if tuple(r["members"]) not in prev_rings]
            self._track_live(self.sim_time)
            txs = []
            for i in range(first_new, w.tx.n):
                txs.append(self._tx_record(i))
            for t in txs[-40:]:
                self.recent_tx.appendleft(t)
            lat = {"features_ms": (t_feat - t0) * 1e3, "scoring_ms": (t_score - t_feat) * 1e3,
                   "total_ms": (time.perf_counter() - t0) * 1e3, "n_tx": len(rows), "rescored": n_rescored,
                   "banks": banks}
            self.latencies.append(lat)
        return {"tx": txs, "alerts": new_alerts, "new_rings": [self.ring_summary(r) for r in new_rings], "latency": lat}

    def _tx_record(self, i: int) -> dict:
        t = self.world.tx
        s, d, mch = int(t.src[i]), int(t.dst[i]), int(t.merchant[i])
        tr = self.tx_risk[i] if i < len(self.tx_risk) else np.nan
        risk = float(tr) if not np.isnan(tr) else max(self.scores[s], self.scores[d] if d >= 0 else 0.0)
        return {"index": i, "src": s, "src_name": self.world.account_name(s), "src_bank": int(self.world.acc_bank[s]),
                "dst": d, "dst_name": self.world.account_name(d) if d >= 0 else None,
                "dst_bank": int(self.world.acc_bank[d]) if d >= 0 else None,
                "merchant": self.world.merchant_names[mch] if mch >= 0 else None,
                "amount": float(t.amount[i]), "ts": float(t.ts[i]), "time": self.fmt_time(t.ts[i]),
                "kind": "transfer" if t.kind[i] == TX_TRANSFER else "merchant", "risk": float(risk),
                "format": self.world.formats[t.fmt[i]] if self.world.formats and t.fmt[i] >= 0 else None,
                "dst_member": bool(d >= 0 and self.world.acc_bank[d] < self.world.n_banks),
                "cross_bank": bool(d >= 0 and self.world.acc_bank[s] != self.world.acc_bank[d]),
                "flagged": bool(risk >= self.tx_threshold), "truth": int(t.label[i])}

    def _alert(self, a: int, ts: float) -> dict:
        from .explain import reasons
        w = self.world
        ring = next((r["id"] for r in self.rings if a in r["members"]), None)
        return {"id": f"AL-{len(self.alerts) + 1:05d}-{a}", "account": a, "name": w.account_name(a),
                "bank": int(w.acc_bank[a]), "bank_name": w.bank_names[w.acc_bank[a]], "score": float(self.scores[a]),
                "ts": ts, "time": self.fmt_time(ts), "reasons": reasons(w, self.X_raw, a)[:3], "ring": ring,
                "truth": int(w.acc_label[a])}

    def _live_positive_stats(self):
        lp = getattr(self.world, "live_pos", None)
        if lp is None:
            return None
        ids = np.flatnonzero(lp & self.world.is_member)
        t = self.world.tx
        seen = np.zeros(self.world.n_accounts, bool)
        seen[t.view("src")] = True
        seen[t.view("dst")[t.view("dst") >= 0]] = True
        return {"total": int(len(ids)), "flagged": int((self.scores[ids] >= self.threshold).sum()),
                "active": int(seen[ids].sum())}

    def _track_live(self, now):
        for r in self.world.rings:
            if not r.get("live") or r.get("activated_ts", 0) > now:
                continue
            info = self.live_rings.setdefault(r["id"], {"ring": r["id"], "pattern": r["pattern"], "members": r["members"],
                                                       "banks": r["banks"], "activated": r["activated_ts"],
                                                       "detected": None, "flagged": 0})
            flagged = int((self.scores[r["members"]] >= self.threshold).sum())
            info["flagged"] = flagged
            if info["detected"] is None and flagged >= 0.5 * len(r["members"]):
                info["detected"] = now
                info["minutes_to_detect"] = (now - r["activated_ts"]) / 60
                self.emit({"type": "ring_detected", "ring": info})

    # ------------------------------------------------------------------ stream simulator
    def start_stream(self, **cfg):
        self.stream_cfg.update({k: v for k, v in cfg.items() if v is not None})
        if self.stream_on:
            return
        self.stream_on = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()

    def stop_stream(self):
        self.stream_on = False

    def inject_ring(self):
        with self.lock:
            ring, rows = activate_live_ring(self.world, self.sim_time)
            if ring is None:
                return None
            # stretch the ring's transactions over simulated time so detection is progressive
            self.pending = sorted(self.pending + rows, key=lambda r: r[4])
            self.emit({"type": "ring_activated", "ring": {"id": ring["id"], "pattern": ring["pattern"],
                                                          "banks": ring["banks"], "size": len(ring["members"])}})
            return ring

    def _stream_loop(self):
        rng = np.random.default_rng(self.seed + 99)
        while self.stream_on:
            t_tick = time.time()
            self.ticks += 1
            if self.world.sleepers and self.ticks % int(self.stream_cfg["ring_every"]) == 3:
                self.inject_ring()
            step = float(self.stream_cfg["speed"])
            if self.world.source == "ibm_aml":
                rows = self._replay(step)
            else:
                rows = normal_traffic(self.world, self.sim_time, int(rng.poisson(self.stream_cfg["rate"])))
                rows = [(s, d, m, a, self.sim_time + (t - self.sim_time) * step, k) for s, d, m, a, t, k in rows]
                self.sim_time += step
            with self.lock:
                due = [r for r in self.pending if r[4] <= self.sim_time]
                self.pending = [r for r in self.pending if r[4] > self.sim_time]
            rows = sorted(rows + due, key=lambda r: r[4])
            try:
                out = self.process(rows)
                self.emit({"type": "batch", "sim_time": self.sim_time, "time": self.fmt_time(self.sim_time), **out,
                           "stats": self.stream_stats()})
            except Exception as e:  # keep streaming; surface the error to the UI
                self.emit({"type": "error", "message": repr(e)})
            time.sleep(max(0.0, 1.0 - (time.time() - t_tick)))

    MAX_REPLAY_BATCH = 400

    def _replay(self, step: float) -> list:
        """Next slice of real held-out transactions (timestamp order), at most MAX_REPLAY_BATCH."""
        fut = self.world.future
        end_t = self.sim_time + step
        rows = []
        while self.replay_pos < len(fut) and fut[self.replay_pos][4] <= end_t and len(rows) < self.MAX_REPLAY_BATCH:
            r = fut[self.replay_pos]
            rows.append((int(r[0]), int(r[1]), int(r[2]), float(r[3]), float(r[4]), int(r[5]), int(r[6]), int(r[7])))
            self.replay_pos += 1
        if len(rows) < self.MAX_REPLAY_BATCH:
            self.sim_time = end_t
        elif rows:
            self.sim_time = rows[-1][4]
        if self.replay_pos >= len(fut):
            self.stream_on = False
            self.emit({"type": "replay_done"})
        return rows

    def stream_stats(self):
        lat = list(self.latencies)
        return {"on": self.stream_on, "ticks": self.ticks, "sim_time": self.sim_time, "time": self.fmt_time(self.sim_time),
                "n_tx": int(self.world.tx.n), "alerts": len(self.alerts), "rings": len(self.rings),
                "latency_ms_p50": float(np.median([l["total_ms"] for l in lat])) if lat else None,
                "latency_ms_p95": float(np.percentile([l["total_ms"] for l in lat], 95)) if lat else None,
                "live_rings": sorted(self.live_rings.values(), key=lambda r: r["activated"], reverse=True)[:30],
                "sleepers_left": len(self.world.sleepers), "source": self.world.source,
                "replay": {"done": self.replay_pos, "total": len(self.world.future)} if self.world.source == "ibm_aml" else None,
                "live_positives": self._live_positive_stats(), "live_tx": dict(self.live_tx),
                "tx_threshold": self.tx_threshold,
                **{k: self.stream_cfg[k] for k in self.stream_cfg}}

    # ------------------------------------------------------------------ retraining (privacy/utility experiments)
    def retrain_federated(self, rounds=25, local_epochs=4, dp_noise=0.0, emb_noise=0.05, exchange=True):
        self.status.update(state="training", progress=0.0, message="Federated retraining")
        try:
            m, hist, priv = train_federated(self.prep, rounds, local_epochs, exchange=exchange, dp_noise=dp_noise,
                                            emb_noise=emb_noise, progress=self._progress)
            ex = exchange_embeddings(m, self.prep, emb_noise) if exchange else None
            res = evaluate(self.prep, predict(m, self.prep, ACTIVE if exchange else "nested_local", ex))
            exp = {"id": len(self.experiments) + 1, "rounds": rounds, "local_epochs": local_epochs, "dp_noise": dp_noise,
                   "emb_noise": emb_noise, "exchange": exchange, **res, "privacy": priv, "history": hist}
            self.experiments.append(exp)
            self.emit({"type": "experiment", "experiment": {k: v for k, v in exp.items() if k != "history"}})
            return exp
        finally:
            self.status.update(state="ready", progress=1.0, message="Ready")

    # ------------------------------------------------------------------ helpers / views
    def fmt_time(self, ts):
        return time.strftime("%Y-%m-%d %H:%M", time.gmtime(self.world.t0_epoch + float(ts)))

    def ring_summary(self, r):
        w = self.world
        return {**{k: r[k] for k in ("id", "size", "banks", "total_flow", "mean_score", "max_score", "n_links",
                                     "device_links", "pattern", "pattern_label", "matched_true_ring", "precision", "pseudonym")},
                "bank_names": [w.bank_name(b) for b in r["banks"]], "n_tx": r.get("n_tx"),
                "member_accounts": r.get("member_accounts", r["size"]), "basis": "transactions" if "n_tx" in r else "accounts",
                "live": bool(r["matched_true_ring"] >= 0 and w.rings[r["matched_true_ring"]].get("live"))}

    def account_summary(self, a: int):
        w = self.world
        return {"id": a, "name": w.account_name(a), "bank": int(w.acc_bank[a]), "bank_name": w.bank_names[w.acc_bank[a]],
                "score": float(self.scores[a]), "flagged": bool(self.scores[a] >= self.threshold),
                "role": w.acc_role[a], "truth": int(w.acc_label[a]), "true_ring": int(w.acc_ring[a]),
                "pseudonym": pseudonym("acc", a),
                "member": bool(w.acc_bank[a] < w.n_banks),
                "split": SPLITS.get(int(self.prep.split[a]), "live") if a < len(self.prep.split) else "live"}
