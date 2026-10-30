"""Behavioural features for accounts, devices and merchants.

Every account feature is computable by the account's own bank from its own ledger (a bank
sees both legs of any transfer that touches its customer). Device sharing counts across banks
come from a private-set-intersection style count (see privacy.py). The feature set adapts to
the data source: synthetic data has devices / merchants / account age; the IBM AML data has
payment formats, currency conversion and external-institution counterparties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DAY, TX_MERCHANT, TX_TRANSFER, World

BASE_FEATURES = [
    "log_in_amount", "log_out_amount", "log_in_count", "log_out_count",
    "log_in_counterparties", "log_out_counterparties", "cross_bank_in_ratio", "cross_bank_out_ratio",
    "pass_through_ratio", "log_holding_minutes", "structuring_ratio", "night_ratio", "amount_cv",
    "round_amount_ratio",
]
SYNTH_FEATURES = ["log_account_age_days", "n_devices", "log_device_sharing", "device_bank_span", "risky_merchant_ratio"]
AML_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
AML_FEATURES = ["log_max_amount", "cross_currency_ratio", "external_ratio", "log_self_transfers", "active_days",
                "fan_in_burst", "fan_out_burst"] + [f"share_{f.lower().replace(' ', '_')}" for f in AML_FORMATS]
DEVICE_FEATURES = ["log_accounts", "bank_span", "new_account_ratio"]
MERCHANT_FEATURES = ["log_customers", "high_risk", "log_volume"]
EDGE_DIM = 6
TX_DIM = 14  # per-transaction attributes for the transaction head
MAX_HOLD_MIN = 7 * 24 * 60.0
STRUCT_LO, STRUCT_HI = 9000, 10000

FEATURE_LABELS = {
    "log_in_amount": "Total inflow", "log_out_amount": "Total outflow",
    "log_in_count": "Incoming transfer count", "log_out_count": "Outgoing transfer count",
    "log_in_counterparties": "Distinct senders", "log_out_counterparties": "Distinct receivers",
    "cross_bank_in_ratio": "Cross-bank inflow share", "cross_bank_out_ratio": "Cross-bank outflow share",
    "pass_through_ratio": "Pass-through ratio", "log_holding_minutes": "Fund holding time",
    "structuring_ratio": "Sub-threshold (structuring) share", "night_ratio": "Night-time activity",
    "log_account_age_days": "Account age", "n_devices": "Devices used",
    "log_device_sharing": "Device sharing", "device_bank_span": "Banks on shared devices",
    "risky_merchant_ratio": "High-risk merchant spend", "amount_cv": "Amount variability",
    "round_amount_ratio": "Round-amount share", "log_max_amount": "Largest transfer",
    "cross_currency_ratio": "Currency-conversion share", "external_ratio": "Non-member counterparty share",
    "log_self_transfers": "Self-transfers (reinvestment)", "active_days": "Active days",
    "fan_in_burst": "Max senders in 24h", "fan_out_burst": "Max receivers in 24h",
    **{f"share_{f.lower().replace(' ', '_')}": f"{f} share" for f in AML_FORMATS},
}


def feature_names(world: World) -> list:
    if world.source == "ibm_aml":
        return BASE_FEATURES + AML_FEATURES
    return BASE_FEATURES + SYNTH_FEATURES


def tx_frame(world: World, accounts=None) -> pd.DataFrame:
    t = world.tx
    df = pd.DataFrame({c: t.view(c) for c in t.cols})
    if accounts is not None:
        acc = np.asarray(sorted(accounts))
        df = df[np.isin(df.src.values, acc) | np.isin(df.dst.values, acc)]
    return df


def device_stats(world: World):
    """Per-device (n_accounts, n_banks, new_account_ratio) -- what the PSI registry would return."""
    D = world.n_devices
    n_acc = np.zeros(D)
    banks = [set() for _ in range(D)]
    new = np.zeros(D)
    now = world.tx.view("ts").max() if world.tx.n else world.horizon
    for a, devs in enumerate(world.acc_devices):
        young = (now - world.acc_open_ts[a]) < 90 * DAY
        for d in devs:
            n_acc[d] += 1
            banks[d].add(int(world.acc_bank[a]))
            new[d] += young
    n_banks = np.array([len(b) for b in banks], dtype=float)
    return n_acc, n_banks, np.divide(new, np.maximum(n_acc, 1))


def device_features(world: World, stats=None) -> np.ndarray:
    n_acc, n_banks, new_ratio = stats if stats is not None else device_stats(world)
    return np.stack([np.log1p(n_acc), n_banks / max(world.n_banks, 1), new_ratio], 1).astype(np.float32).reshape(-1, 3)


def _bank_key(world: World) -> np.ndarray:
    return world.acc_code if world.acc_code is not None else world.acc_bank


def account_features(world: World, accounts=None, dev_stats=None) -> np.ndarray:
    """Raw (unnormalised) account features. If `accounts` is given, only those rows are returned."""
    names = feature_names(world)
    col = {n: i for i, n in enumerate(names)}
    N = world.n_accounts
    ids = np.arange(N) if accounts is None else np.asarray(sorted(accounts))
    df = tx_frame(world, None if accounts is None else ids)
    code = _bank_key(world)
    F = np.zeros((len(ids), len(names)), np.float32)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    now = world.tx.view("ts").max() if world.tx.n else world.horizon
    idset = np.zeros(N, bool)
    idset[ids] = True

    tr = df[df.kind == TX_TRANSFER]
    tr = tr.assign(cross=(code[tr.src.values] != code[tr.dst.values]).astype(float),
                   night=(((tr.ts.values % DAY) / 3600) < 6).astype(float),
                   struct=((tr.amount.values >= STRUCT_LO) & (tr.amount.values < STRUCT_HI)).astype(float),
                   rnd=((np.round(tr.amount.values, 2) % 100) == 0).astype(float))
    out = tr[idset[tr.src.values]].groupby("src").agg(
        amt=("amount", "sum"), cnt=("amount", "size"), uniq=("dst", "nunique"), cross=("cross", "sum"),
        night=("night", "sum"), struct=("struct", "sum"), rnd=("rnd", "sum"), mean=("amount", "mean"),
        std=("amount", "std"), mx=("amount", "max"))
    inn = tr[idset[tr.dst.values]].groupby("dst").agg(
        amt=("amount", "sum"), cnt=("amount", "size"), uniq=("src", "nunique"), cross=("cross", "sum"),
        night=("night", "sum"), struct=("struct", "sum"), mx=("amount", "max"))

    def put(series, name, fn=lambda v: v):
        if len(series):
            F[pos[series.index].values, col[name]] = fn(series.values)

    put(inn.amt, "log_in_amount", np.log1p)
    put(out.amt, "log_out_amount", np.log1p)
    put(inn.cnt, "log_in_count", np.log1p)
    put(out.cnt, "log_out_count", np.log1p)
    put(inn.uniq, "log_in_counterparties", np.log1p)
    put(out.uniq, "log_out_counterparties", np.log1p)
    put(inn.cross / inn.cnt, "cross_bank_in_ratio")
    put(out.cross / out.cnt, "cross_bank_out_ratio")
    put((out["std"].fillna(0) / out["mean"].clip(lower=1e-6)).clip(0, 5), "amount_cv")
    put(out.rnd / out.cnt, "round_amount_ratio")

    in_amt = np.zeros(len(ids)); out_amt = np.zeros(len(ids))
    tot_cnt = np.zeros(len(ids)); night = np.zeros(len(ids)); struct = np.zeros(len(ids))
    if len(inn):
        p = pos[inn.index].values
        in_amt[p] = inn.amt.values; tot_cnt[p] += inn.cnt.values; night[p] += inn.night.values; struct[p] += inn.struct.values
    if len(out):
        p = pos[out.index].values
        out_amt[p] = out.amt.values; tot_cnt[p] += out.cnt.values; night[p] += out.night.values; struct[p] += out.struct.values
    F[:, col["pass_through_ratio"]] = np.minimum(in_amt, out_amt) / np.maximum(np.maximum(in_amt, out_amt), 1)
    F[:, col["night_ratio"]] = night / np.maximum(tot_cnt, 1)
    F[:, col["structuring_ratio"]] = struct / np.maximum(tot_cnt, 1)

    # holding time: for each outgoing transfer, minutes since the latest incoming transfer
    hold = np.full(len(ids), MAX_HOLD_MIN)
    if len(tr):
        o = tr[idset[tr.src.values]][["src", "ts"]].rename(columns={"src": "acc"}).sort_values("ts")
        i = tr[idset[tr.dst.values]][["dst", "ts"]].rename(columns={"dst": "acc", "ts": "in_ts"}).sort_values("in_ts")
        if len(o) and len(i):
            m = pd.merge_asof(o, i, left_on="ts", right_on="in_ts", by="acc", direction="backward", tolerance=7 * DAY)
            m = m.dropna()
            if len(m):
                g = ((m.ts - m.in_ts) / 60).groupby(m.acc).median()
                hold[pos[g.index].values] = g.values
    F[:, col["log_holding_minutes"]] = np.log1p(hold)

    if world.source == "ibm_aml":
        _aml_features(world, tr, idset, pos, F, col, out, inn, in_amt, out_amt, tot_cnt, ids)
    else:
        _synthetic_features(world, df, ids, pos, F, col, out_amt, now, dev_stats)
    return F


def _aml_features(world, tr, idset, pos, F, col, out, inn, in_amt, out_amt, tot_cnt, ids):
    mx = np.zeros(len(ids))
    if len(out):
        mx[pos[out.index].values] = out.mx.values
    if len(inn):
        p = pos[inn.index].values
        mx[p] = np.maximum(mx[p], inn.mx.values)
    F[:, col["log_max_amount"]] = np.log1p(mx)
    ext = (world.acc_bank >= world.n_banks)
    legs = pd.concat([
        pd.DataFrame({"acc": tr.src.values, "x": tr.xcur.values, "e": ext[tr.dst.values], "f": tr.fmt.values,
                      "day": (tr.ts.values // DAY), "cp": tr.dst.values, "dir": 1}),
        pd.DataFrame({"acc": tr.dst.values, "x": tr.xcur.values, "e": ext[tr.src.values], "f": tr.fmt.values,
                      "day": (tr.ts.values // DAY), "cp": tr.src.values, "dir": 0}),
    ])
    legs = legs[idset[legs.acc.values]]
    if len(legs):
        g = legs.groupby("acc")
        p = pos[g.size().index].values
        F[p, col["cross_currency_ratio"]] = g.x.mean().values
        F[p, col["external_ratio"]] = g.e.mean().values
        F[p, col["active_days"]] = g.day.nunique().values
        sh = pd.crosstab(legs.acc, legs.f, normalize="index")
        for k, f in enumerate(AML_FORMATS):
            if k in sh.columns:
                F[pos[sh.index].values, col[f"share_{f.lower().replace(' ', '_')}"]] = sh[k].values
        for d, name in ((0, "fan_in_burst"), (1, "fan_out_burst")):
            b = legs[legs.dir == d].groupby(["acc", "day"]).cp.nunique().groupby(level=0).max()
            if len(b):
                F[pos[b.index].values, col[name]] = np.log1p(b.values)
    self_cnt = getattr(world, "self_cnt", None)
    if self_cnt is not None:
        F[:, col["log_self_transfers"]] = np.log1p(self_cnt[ids])


def _synthetic_features(world, df, ids, pos, F, col, out_amt, now, dev_stats):
    F[:, col["log_account_age_days"]] = np.log1p(np.maximum(now - world.acc_open_ts[ids], 0) / DAY)
    n_acc, n_banks, _ = dev_stats if dev_stats is not None else device_stats(world)
    for k, a in enumerate(ids):
        devs = world.acc_devices[a]
        F[k, col["n_devices"]] = len(devs)
        if devs:
            F[k, col["log_device_sharing"]] = np.log1p(max(n_acc[d] for d in devs) - 1)
            F[k, col["device_bank_span"]] = max(n_banks[d] for d in devs)
    me = df[(df.kind == TX_MERCHANT) & df.src.isin(ids)]
    if len(me):
        risky = world.merchant_risk[me.merchant.values].astype(float)
        g = pd.DataFrame({"src": me.src.values, "amt": me.amount.values, "r": risky * me.amount.values}).groupby("src").sum()
        tot = g.amt.values + out_amt[pos[g.index].values]
        F[pos[g.index].values, col["risky_merchant_ratio"]] = g.r.values / np.maximum(tot, 1)


class Normalizer:
    """Consortium-wide standardisation (means/stds could be obtained by secure aggregation of sums)."""

    def __init__(self, X: np.ndarray):
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6

    def __call__(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


def merchant_features(world: World, df: pd.DataFrame) -> np.ndarray:
    M = world.n_merchants
    if M == 0:
        return np.zeros((0, 3), np.float32)
    me = df[df.kind == TX_MERCHANT]
    cust = me.groupby("merchant").src.nunique().reindex(range(M), fill_value=0).values
    vol = me.groupby("merchant").amount.sum().reindex(range(M), fill_value=0).values
    return np.stack([np.log1p(cust), world.merchant_risk.astype(float), np.log1p(vol / 1000)], 1).astype(np.float32)


def tx_features(world: World, idx: np.ndarray) -> np.ndarray:
    """Per-transaction attributes: log amount, payment rail one-hot (7), currency conversion,
    time of day (sin, cos), cross-institution flag, member flags of both endpoints."""
    t = world.tx
    idx = np.asarray(idx, np.int64)
    src, dst = t.src[idx], t.dst[idx]
    dst_ok = np.maximum(dst, 0)
    code = _bank_key(world)
    member = world.is_member
    X = np.zeros((len(idx), TX_DIM), np.float32)
    X[:, 0] = np.log1p(t.amount[idx]) / 10
    f = t.fmt[idx]
    ok = (f >= 0) & (f < 7)
    X[np.flatnonzero(ok), 1 + f[ok]] = 1
    X[:, 8] = t.xcur[idx]
    hour = (t.ts[idx] % DAY) / 3600
    X[:, 9] = np.sin(hour / 24 * 2 * np.pi)
    X[:, 10] = np.cos(hour / 24 * 2 * np.pi)
    X[:, 11] = (code[src] != code[dst_ok]) & (dst >= 0)
    X[:, 12] = member[src]
    X[:, 13] = member[dst_ok] & (dst >= 0)
    return X
