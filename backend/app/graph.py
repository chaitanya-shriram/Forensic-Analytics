"""Heterogeneous graph views and nested (rooted) subgraph extraction.

A *view* is the part of the consortium graph one party can see:
  * central view  -- every account (upper bound, needs pooled data; used only for comparison)
  * bank view     -- the bank's own accounts, the devices and merchants they use, and
                     *ghost* nodes for counterparty accounts held at other banks. A bank
                     knows a ghost only through the transfers it took part in.

Node index space of a view: [accounts (own first, then ghosts) | devices | merchants].

Relation types (inner, heterogeneous graph):
  0 transfer src->dst   1 transfer dst->src   2 account->device  3 device->account
  4 account->merchant   5 merchant->account
Relation types (outer, account-level graph):
  0 transfer src->dst   1 transfer dst->src   2 shared device (symmetric)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data import TX_MERCHANT, TX_TRANSFER, World
from .features import EDGE_DIM, merchant_features, tx_features, tx_frame

N_INNER_REL = 6
N_OUTER_REL = 3
NODE_OWN, NODE_GHOST, NODE_DEVICE, NODE_MERCHANT = 0, 1, 2, 3
HIGH_RISK_FORMATS = [1, 2]  # IBM AML: Bitcoin, Cash


@dataclass
class GraphView:
    bank: int | None
    acc_ids: np.ndarray          # global ids of view accounts (own first, then ghosts)
    n_own: int
    dev_ids: np.ndarray          # global device ids
    mer_ids: np.ndarray          # global merchant ids
    x_acc: torch.Tensor          # [n_own, F_acc] normalised features of own accounts
    x_dev: torch.Tensor
    x_mer: torch.Tensor
    edge_index: torch.Tensor     # [2, E] inner heterogeneous edges (view node space)
    edge_type: torch.Tensor
    edge_attr: torch.Tensor
    acc_edge_index: torch.Tensor  # [2, E'] outer account-level edges (account index space)
    acc_edge_type: torch.Tensor
    acc_edge_attr: torch.Tensor
    csr_ptr: np.ndarray          # CSR over inner edges by source node (for subgraph extraction)
    csr_dst: np.ndarray
    csr_eid: np.ndarray
    e_src: np.ndarray | None = None  # host copies of edge_index for subgraph extraction
    e_dst: np.ndarray | None = None
    tx_ids: np.ndarray | None = None   # global ids of the transfers this bank sees
    tx_src: torch.Tensor | None = None  # view-local account index of each transfer's endpoints
    tx_dst: torch.Tensor | None = None
    x_tx: torch.Tensor | None = None    # per-transaction attributes (features.tx_features)

    @property
    def n_acc(self):
        return len(self.acc_ids)

    @property
    def n_ghost(self):
        return len(self.acc_ids) - self.n_own

    @property
    def n_nodes(self):
        return len(self.acc_ids) + len(self.dev_ids) + len(self.mer_ids)

    def node_type(self) -> np.ndarray:
        t = np.empty(self.n_nodes, np.int64)
        t[: self.n_own] = NODE_OWN
        t[self.n_own: self.n_acc] = NODE_GHOST
        t[self.n_acc: self.n_acc + len(self.dev_ids)] = NODE_DEVICE
        t[self.n_acc + len(self.dev_ids):] = NODE_MERCHANT
        return t

    def to(self, device):
        for k in ("x_acc", "x_dev", "x_mer", "edge_index", "edge_type", "edge_attr",
                  "acc_edge_index", "acc_edge_type", "acc_edge_attr", "tx_src", "tx_dst", "x_tx"):
            if getattr(self, k) is not None:
                setattr(self, k, getattr(self, k).to(device))
        return self

    def local_index(self) -> dict:
        return {int(g): i for i, g in enumerate(self.acc_ids)}


def build_view(world: World, X_acc_norm: np.ndarray, X_dev: np.ndarray, bank: int | None = None,
               df: pd.DataFrame | None = None) -> GraphView:
    N = world.n_accounts
    if df is None:
        df = tx_frame(world)
    own = np.flatnonzero(world.acc_bank < world.n_banks) if bank is None else np.flatnonzero(world.acc_bank == bank)
    own_mask = np.zeros(N, bool)
    own_mask[own] = True

    tr = df[df.kind == TX_TRANSFER]
    tr = tr[own_mask[tr.src.values] | own_mask[tr.dst.values]]
    me = df[(df.kind == TX_MERCHANT) & own_mask[df.src.values]]

    # account index: own first, then ghosts (counterparties at other banks)
    counter = np.unique(np.concatenate([tr.src.values, tr.dst.values])) if len(tr) else np.zeros(0, np.int64)
    ghosts = counter[~own_mask[counter]]
    acc_ids = np.concatenate([own, ghosts]).astype(np.int64)
    acc_pos = np.full(N, -1, np.int64)
    acc_pos[acc_ids] = np.arange(len(acc_ids))

    # devices used by own accounts
    pairs = [(a, d) for a in own for d in world.acc_devices[a]]
    dev_ids = np.unique(np.array([d for _, d in pairs], np.int64)) if pairs else np.zeros(0, np.int64)
    dev_pos = {int(d): i for i, d in enumerate(dev_ids)}
    mer_ids = np.unique(me.merchant.values).astype(np.int64)
    mer_pos = np.full(world.n_merchants, -1, np.int64)
    mer_pos[mer_ids] = np.arange(len(mer_ids))
    A, D = len(acc_ids), len(dev_ids)

    # --- aggregated transfer edges
    bank_arr = world.acc_code if world.acc_code is not None else world.acc_bank
    g = tr.assign(struct=((tr.amount >= 9000) & (tr.amount < 10000)).astype(float),
                  risky=np.isin(tr.fmt.values, HIGH_RISK_FORMATS).astype(float)).groupby(["src", "dst"]).agg(
        amt=("amount", "sum"), cnt=("amount", "size"), struct=("struct", "mean"), xcur=("xcur", "mean"),
        risky=("risky", "mean")).reset_index()
    ts = acc_pos[g.src.values]
    td = acc_pos[g.dst.values]
    tattr = np.stack([np.log1p(g.amt.values) / 10, np.log1p(g.cnt.values),
                      (bank_arr[g.src.values] != bank_arr[g.dst.values]).astype(float), g.struct.values,
                      g.xcur.values, g.risky.values], 1).reshape(-1, EDGE_DIM)

    # --- account-device edges
    if pairs:
        ad_a = np.array([acc_pos[a] for a, _ in pairs])
        ad_d = np.array([A + dev_pos[d] for _, d in pairs])
    else:
        ad_a = ad_d = np.zeros(0, np.int64)
    dattr = np.zeros((len(ad_a), EDGE_DIM))

    # --- account-merchant edges
    mg = me.groupby(["src", "merchant"]).amount.agg(["sum", "size"]).reset_index()
    am_a = acc_pos[mg.src.values]
    am_m = A + D + mer_pos[mg.merchant.values]
    mattr = np.zeros((len(mg), EDGE_DIM))
    if len(mg):
        mattr[:, 0] = np.log1p(mg["sum"].values) / 10
        mattr[:, 1] = np.log1p(mg["size"].values)
        mattr[:, 5] = world.merchant_risk[mg.merchant.values]

    src = np.concatenate([ts, td, ad_a, ad_d, am_a, am_m])
    dst = np.concatenate([td, ts, ad_d, ad_a, am_m, am_a])
    etype = np.concatenate([np.full(len(ts), 0), np.full(len(ts), 1), np.full(len(ad_a), 2), np.full(len(ad_a), 3),
                            np.full(len(am_a), 4), np.full(len(am_a), 5)])
    eattr = np.concatenate([tattr, tattr, dattr, dattr, mattr, mattr]).astype(np.float32)

    order = np.argsort(src, kind="stable")
    n_nodes = A + D + len(mer_ids)
    csr_ptr = np.zeros(n_nodes + 1, np.int64)
    np.add.at(csr_ptr, src + 1, 1)
    csr_ptr = np.cumsum(csr_ptr)

    # --- outer account-level graph: transfers + shared-device links among view accounts
    dev_users: dict[int, list] = {}
    for a, d in pairs:
        dev_users.setdefault(d, []).append(acc_pos[a])
    sd_s, sd_d, sd_w = [], [], []
    for d, us in dev_users.items():
        if 1 < len(us) <= 30:
            for i in us:
                for j in us:
                    if i != j:
                        sd_s.append(i); sd_d.append(j); sd_w.append(len(us))
    sd_attr = np.zeros((len(sd_s), EDGE_DIM))
    if sd_s:
        sd_attr[:, 1] = np.log1p(np.array(sd_w, float))
    o_src = np.concatenate([ts, td, np.array(sd_s, np.int64)])
    o_dst = np.concatenate([td, ts, np.array(sd_d, np.int64)])
    o_type = np.concatenate([np.zeros(len(ts)), np.ones(len(ts)), np.full(len(sd_s), 2)])
    o_attr = np.concatenate([tattr, tattr, sd_attr]).astype(np.float32)

    mer_x = merchant_features(world, df[df.src.isin(own)] if bank is not None else df)[mer_ids]
    return GraphView(
        bank=bank, acc_ids=acc_ids, n_own=len(own), dev_ids=dev_ids, mer_ids=mer_ids,
        x_acc=torch.from_numpy(X_acc_norm[own]), x_dev=torch.from_numpy(X_dev[dev_ids]),
        x_mer=torch.from_numpy(mer_x),
        edge_index=torch.from_numpy(np.stack([src, dst]).astype(np.int64)),
        edge_type=torch.from_numpy(etype.astype(np.int64)), edge_attr=torch.from_numpy(eattr),
        acc_edge_index=torch.from_numpy(np.stack([o_src, o_dst]).astype(np.int64)),
        acc_edge_type=torch.from_numpy(o_type.astype(np.int64)), acc_edge_attr=torch.from_numpy(o_attr),
        csr_ptr=csr_ptr, csr_dst=dst[order], csr_eid=order, e_src=src.astype(np.int64), e_dst=dst.astype(np.int64),
        tx_ids=tr.index.values.astype(np.int64),
        tx_src=torch.from_numpy(acc_pos[tr.src.values].astype(np.int64)),
        tx_dst=torch.from_numpy(acc_pos[tr.dst.values].astype(np.int64)),
        x_tx=torch.from_numpy(tx_features(world, tr.index.values)),
    )


@dataclass
class SubgraphBatch:
    """Disjoint union of rooted k-hop subgraphs, one per root (the 'nested' level)."""
    roots: np.ndarray            # view account indices of roots
    node_idx: torch.Tensor       # [S] view node index for every subgraph node copy
    dist: torch.Tensor           # [S] hop distance to the subgraph root
    batch: torch.Tensor          # [S] subgraph id
    root_pos: torch.Tensor       # [B] position of each root inside node_idx
    edge_index: torch.Tensor     # [2, Es] batched local edges
    edge_type: torch.Tensor
    edge_attr: torch.Tensor
    edge_view_id: torch.Tensor   # [Es] id of the corresponding view edge (for explanations)

    def to(self, device):
        for k in ("node_idx", "dist", "batch", "root_pos", "edge_index", "edge_type", "edge_attr", "edge_view_id"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def extract_subgraphs(view: GraphView, roots=None, hops: int = 2, max_hop1: int = 12, max_per_node: int = 4,
                      max_nodes: int = 36, seed: int = 0) -> SubgraphBatch:
    roots = np.arange(view.n_own) if roots is None else np.asarray(roots, np.int64)
    ptr, cdst, ceid = view.csr_ptr, view.csr_dst, view.csr_eid
    e_src, e_dst = view.e_src, view.e_dst
    mark = np.full(view.n_nodes, -1, np.int64)
    mer_start = view.n_acc + len(view.dev_ids)
    node_l, dist_l, batch_l, rootpos_l, es_l, ed_l, eid_l = [], [], [], [], [], [], []
    offset = 0
    for b, r in enumerate(roots):
        rng = np.random.default_rng(seed * 1_000_003 + int(view.acc_ids[r]))
        nodes = [int(r)]
        dist = [0]
        mark[r] = 0
        frontier = [int(r)]
        for h in range(1, hops + 1):
            nxt = []
            cap = max_hop1 if h == 1 else max_per_node
            for u in frontier:
                if u >= mer_start and u != r:
                    continue  # merchants are leaves: expanding them only adds unrelated shoppers
                nb = [x for x in dict.fromkeys(cdst[ptr[u]:ptr[u + 1]].tolist()) if mark[x] < 0]
                if len(nb) > cap:
                    nb = rng.choice(nb, cap, replace=False).tolist()
                for v in nb:
                    if len(nodes) >= max_nodes:
                        break
                    mark[v] = len(nodes)
                    nodes.append(int(v)); dist.append(h); nxt.append(int(v))
            frontier = nxt
        nodes_a = np.array(nodes, np.int64)
        # induced edges
        starts, ends = ptr[nodes_a], ptr[nodes_a + 1]
        eids = np.concatenate([ceid[s:e] for s, e in zip(starts, ends)]) if len(nodes_a) else np.zeros(0, np.int64)
        keep = eids[mark[e_dst[eids]] >= 0]
        es_l.append(mark[e_src[keep]] + offset)
        ed_l.append(mark[e_dst[keep]] + offset)
        eid_l.append(keep)
        node_l.append(nodes_a); dist_l.append(dist); batch_l.append(np.full(len(nodes_a), b)); rootpos_l.append(offset)
        mark[nodes_a] = -1
        offset += len(nodes_a)
    eid = np.concatenate(eid_l) if eid_l else np.zeros(0, np.int64)
    eid_t = torch.from_numpy(eid.astype(np.int64))
    return SubgraphBatch(
        roots=roots,
        node_idx=torch.from_numpy(np.concatenate(node_l)), dist=torch.tensor(np.concatenate(dist_l), dtype=torch.long),
        batch=torch.from_numpy(np.concatenate(batch_l).astype(np.int64)), root_pos=torch.tensor(rootpos_l, dtype=torch.long),
        edge_index=torch.from_numpy(np.stack([np.concatenate(es_l), np.concatenate(ed_l)]).astype(np.int64)),
        edge_type=view.edge_type.cpu()[eid_t], edge_attr=view.edge_attr.cpu()[eid_t], edge_view_id=eid_t,
    )


def extract_chunks(view: GraphView, roots=None, chunk: int = 2048, **kw) -> list:
    """Rooted subgraphs in chunks of `chunk` roots (encoded one chunk at a time)."""
    roots = np.arange(view.n_own) if roots is None else np.asarray(roots, np.int64)
    return [extract_subgraphs(view, roots[i:i + chunk], **kw) for i in range(0, max(len(roots), 1), chunk)] \
        if len(roots) else [extract_subgraphs(view, roots, **kw)]


def subs_to(subs, device):
    if isinstance(subs, (list, tuple)):
        return [s.to(device) for s in subs]
    return subs.to(device)
