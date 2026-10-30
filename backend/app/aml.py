"""IBM Transactions for Anti-Money-Laundering (AMLworld, HI-Small) loader and cleaner.

Source: Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering
Models", NeurIPS 2023 Datasets & Benchmarks. Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml.
It is the standard public benchmark with *institution ids on both sides of every transfer* and
labelled laundering typologies (fan-in, fan-out, cycle, scatter-gather, gather-scatter, stack,
bipartite, random), which is exactly what cross-bank ring detection needs.

Consortium construction:
  * the K largest institutions (by number of accounts) form the consortium -- each one is a
    federated client that owns its customers' accounts;
  * every other institution's account that transacts with a member appears as an *external*
    node (acc_bank == K): visible only through the member's own ledger;
  * transactions before `cutoff_day` are history (training / graph); later ones touching a
    member are held out and replayed by the live stream in timestamp order.

Cleaning:
  * drops exact duplicate rows;
  * converts every amount to USD with rates inferred from the dataset's own cross-currency
    transactions (median paid/received ratio against USD, falling back through EUR);
  * drops self-transfers (Reinvestment into the same account) from the graph but keeps them as
    an account feature;
  * keeps payment format and currency conversion as transaction attributes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DAY, TX_TRANSFER, TxStore, World

KAGGLE_ID = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
PATTERN_MAP = {
    "FAN-OUT": "fan_out", "FAN-IN": "fan_in", "CYCLE": "cycle", "SCATTER-GATHER": "scatter_gather",
    "GATHER-SCATTER": "gather_scatter", "STACK": "stack", "BIPARTITE": "bipartite", "RANDOM": "random",
}
T0 = pd.Timestamp("2022-09-01")
CACHE = Path(os.environ.get("FRAUD_DATA_CACHE", Path(__file__).resolve().parent.parent / "data_cache"))


def dataset_dir(variant: str = "HI-Small") -> Path:
    env = os.environ.get("AML_DATA_DIR")
    if env:
        return Path(env)
    import kagglehub  # downloads once into ~/.cache/kagglehub (public dataset, no login needed)
    for f in (f"{variant}_Patterns.txt", f"{variant}_Trans.csv"):
        p = Path(kagglehub.dataset_download(KAGGLE_ID, path=f))
    return p.parent


def read_transactions(d: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(d / f"{variant}_Trans.csv", dtype=str)
    df.columns = ["ts", "from_bank", "from_acc", "to_bank", "to_acc", "amt_recv", "cur_recv",
                  "amt_paid", "cur_paid", "fmt", "label"]
    n0 = len(df)
    df = df.drop_duplicates()
    df["amt_recv"] = df.amt_recv.astype(float)
    df["amt_paid"] = df.amt_paid.astype(float)
    df["label"] = df.label.astype(np.int8)
    df["t"] = (pd.to_datetime(df.ts, format="%Y/%m/%d %H:%M") - T0).dt.total_seconds()
    df["src"] = df.from_bank + ":" + df.from_acc
    df["dst"] = df.to_bank + ":" + df.to_acc
    df.attrs["dropped_duplicates"] = n0 - len(df)
    return df


def usd_rates(df: pd.DataFrame) -> dict:
    """Exchange rates (USD per unit) inferred from the dataset's cross-currency rows."""
    x = df[(df.cur_recv != df.cur_paid) & (df.amt_paid > 0) & (df.amt_recv > 0)]
    pair = (x.amt_recv / x.amt_paid).groupby([x.cur_paid, x.cur_recv]).median()
    rates = {"US Dollar": 1.0}
    for _ in range(3):  # propagate through intermediate currencies
        for (paid, recv), r in pair.items():  # 1 paid = r recv
            if recv in rates and paid not in rates:
                rates[paid] = r * rates[recv]
            elif paid in rates and recv not in rates:
                rates[recv] = rates[paid] / r
    for c in pd.unique(pd.concat([df.cur_paid, df.cur_recv])):
        rates.setdefault(c, 1.0)
    return rates


def read_patterns(d: Path, variant: str) -> list:
    text = (d / f"{variant}_Patterns.txt").read_text()
    out = []
    for blk in text.split("BEGIN LAUNDERING ATTEMPT - ")[1:]:
        lines = blk.strip().splitlines()
        head = lines[0]
        typ = head.split(":")[0].strip()
        accs, ts = [], []
        for line in lines[1:]:
            if line.startswith("END"):
                break
            p = line.split(",")
            accs += [p[1] + ":" + p[2], p[3] + ":" + p[4]]
            ts.append((pd.Timestamp(p[0].replace("/", "-")) - T0).total_seconds())
        out.append({"pattern": PATTERN_MAP.get(typ, typ.lower()), "detail": head.split(":", 1)[-1].strip(),
                    "accounts": list(dict.fromkeys(accs)), "start": min(ts), "end": max(ts)})
    return out


def load_ibm_aml(n_members: int = 8, cutoff_day: float = 8.0, variant: str = "HI-Small",
                 use_cache: bool = True) -> World:
    tag = f"{variant}_k{n_members}_cut{cutoff_day:g}_v2"
    cache = CACHE / f"{tag}.npz"
    if use_cache and cache.exists():
        return _from_cache(cache)
    t_start = time.time()
    d = dataset_dir(variant)
    df = read_transactions(d, variant)
    rates = usd_rates(df)
    df["usd"] = df.amt_paid * df.cur_paid.map(rates)
    report = {"rows": int(len(df) + df.attrs["dropped_duplicates"]), "dropped_duplicates": int(df.attrs["dropped_duplicates"]),
              "rates_usd": {k: round(v, 6) for k, v in rates.items()}}

    # ---- consortium = K institutions with the most accounts
    acc_bank_code = pd.concat([df[["src", "from_bank"]].set_axis(["acc", "bank"], axis=1),
                               df[["dst", "to_bank"]].set_axis(["acc", "bank"], axis=1)]).drop_duplicates("acc")
    members = acc_bank_code.bank.value_counts().index[:n_members].tolist()
    member_idx = {b: i for i, b in enumerate(members)}
    own = acc_bank_code[acc_bank_code.bank.isin(members)]
    touch = df.src.isin(own.acc) | df.dst.isin(own.acc)
    df = df[touch]
    selfx = df.src == df.dst
    report["self_transfers"] = int(selfx.sum())

    # account index: members first (grouped by bank), then external counterparties
    own = own.assign(m=own.bank.map(member_idx)).sort_values(["m", "acc"])
    ext = pd.unique(pd.concat([df.src, df.dst]))
    ext = np.setdiff1d(ext, own.acc.values)
    names = np.concatenate([own.acc.values, ext])
    idx = pd.Series(np.arange(len(names)), index=names)
    code_of = acc_bank_code.set_index("acc").bank
    acc_code = pd.factorize(code_of.reindex(names).values)[0]
    acc_bank = np.concatenate([own.m.values, np.full(len(ext), n_members)]).astype(np.int64)

    # labels: laundering involvement before / after the cutoff
    cut = cutoff_day * DAY
    lab = df[df.label == 1]
    hist_l = pd.unique(pd.concat([lab.src[lab.t < cut], lab.dst[lab.t < cut]]))
    fut_l = pd.unique(pd.concat([lab.src[lab.t >= cut], lab.dst[lab.t >= cut]]))
    N = len(names)
    acc_label = np.zeros(N, np.int64)
    acc_label[idx[hist_l].values] = 1
    live_pos = np.zeros(N, bool)
    live_pos[idx[fut_l].values] = True
    live_pos &= acc_label == 0

    # patterns -> rings (owned members only are what the consortium can find)
    pats = read_patterns(d, variant)
    acc_ring = np.full(N, -1, np.int64)
    rings = []
    for p in pats:
        accs = [idx[a] for a in p["accounts"] if a in idx.index]
        owned = [int(a) for a in accs if acc_bank[a] < n_members]
        if not owned:
            continue
        rid = len(rings)
        rings.append({"id": rid, "pattern": p["pattern"], "detail": p["detail"], "members": owned,
                      "all_members": [int(a) for a in accs], "banks": sorted({int(acc_bank[a]) for a in owned}),
                      "n_institutions": len({acc_code[a] for a in accs}), "live": p["start"] >= cut,
                      "start": p["start"], "end": p["end"], "activated_ts": p["start"]})
        for a in owned:
            if acc_ring[a] < 0:
                acc_ring[a] = rid
    # members of rings that launder only after the cutoff are "live"
    fmt_code = {f: i for i, f in enumerate(FORMATS)}
    g = df[~selfx]
    src = idx[g.src].values
    dst = idx[g.dst].values
    cols = dict(src=src, dst=dst, merchant=np.full(len(g), -1), amount=g.usd.values, ts=g.t.values,
                kind=np.zeros(len(g), np.int8), fmt=g.fmt.map(fmt_code).fillna(-1).astype(np.int8).values,
                xcur=(g.cur_paid != g.cur_recv).astype(np.int8).values, label=g.label.values.astype(np.int8))
    hist = cols["ts"] < cut
    order_h = np.argsort(cols["ts"][hist], kind="stable")
    tx = TxStore.from_arrays(**{k: v[hist][order_h] for k, v in cols.items()})
    fut = ~hist
    order_f = np.argsort(cols["ts"][fut], kind="stable")
    future = np.stack([cols[k][fut][order_f].astype(float) for k in TxStore.cols], 1)
    # self-transfers per account (reinvestment) as a side feature
    sx = df[selfx & (df.t < cut)]
    self_cnt = np.zeros(N)
    np.add.at(self_cnt, idx[sx.src].values, 1)

    report.update({"members": members, "n_accounts": N, "n_member_accounts": int((acc_bank < n_members).sum()),
                   "n_external_accounts": int(len(ext)), "history_tx": int(tx.n), "future_tx": int(len(future)),
                   "positives_history": int(acc_label[acc_bank < n_members].sum()),
                   "live_positives": int(live_pos[acc_bank < n_members].sum()), "rings": len(rings),
                   "cutoff_day": cutoff_day, "load_s": round(time.time() - t_start, 1),
                   "laundering_tx_history": int((lab.t < cut).sum()), "laundering_tx_future": int((lab.t >= cut).sum())})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, names=names.astype(str), acc_bank=acc_bank, acc_code=acc_code, acc_label=acc_label,
                        live_pos=live_pos, acc_ring=acc_ring, future=future, self_cnt=self_cnt,
                        **{f"tx_{c}": tx.view(c) for c in TxStore.cols},
                        meta=np.array(json.dumps({"rings": rings, "report": report, "members": members})))
    return _from_cache(cache)


def _from_cache(path: Path) -> World:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    members = meta["members"]
    N = len(z["names"])
    tx = TxStore.from_arrays(**{c: z[f"tx_{c}"] for c in TxStore.cols})
    names = [n.replace(":", "·") for n in z["names"].tolist()]
    w = World(
        n_banks=len(members), bank_names=[f"Bank {b}" for b in members] + ["External institutions"],
        acc_bank=z["acc_bank"], acc_open_ts=np.full(N, -365 * DAY), acc_label=z["acc_label"].copy(),
        acc_ring=z["acc_ring"], acc_role=["customer"] * N, acc_devices=[[] for _ in range(N)], n_devices=0,
        merchant_risk=np.zeros(0, np.int64), merchant_names=[], tx=tx, rings=meta["rings"],
        horizon=float(tx.view("ts").max()), source="ibm_aml", t0_epoch=float(T0.timestamp()),
        acc_code=z["acc_code"], acc_names=names, future=z["future"].tolist(), formats=FORMATS,
        meta={**meta["report"], "member_codes": members},
    )
    w.live_pos = z["live_pos"]
    w.self_cnt = z["self_cnt"]
    for a in np.flatnonzero(w.acc_label == 1):
        w.acc_role[a] = "launderer"
    return w
