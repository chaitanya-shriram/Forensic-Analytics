"""Nested Heterogeneous Graph Neural Network (pure PyTorch).

Two-level ("nested") architecture, after Zhang & Li, *Nested Graph Neural Networks* (NeurIPS 2021),
extended to heterogeneous financial graphs and to a federated, cross-bank setting:

  inner GNN : for every account a rooted k-hop heterogeneous subgraph (accounts, ghosts,
              devices, merchants) is encoded with relation-aware attention layers and pooled
              into a subgraph embedding z_v. This captures local motifs (fan-in/out, chains,
              device farms) that a plain message-passing GNN cannot distinguish.
  outer GNN : relation-aware attention over the account-level graph whose node features are
              the subgraph embeddings. Ghost accounts (customers of other banks) receive
              privacy-protected embeddings exchanged through the consortium, which is how
              information crosses bank boundaries without sharing raw data.

Attention coefficients of both levels are returned for explanations.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .features import EDGE_DIM, TX_DIM

from .graph import N_INNER_REL, N_OUTER_REL, NODE_DEVICE, NODE_GHOST, NODE_MERCHANT, NODE_OWN, GraphView, SubgraphBatch


def scatter_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    H = score.shape[1]
    idx = index.unsqueeze(1).expand(-1, H)
    mx = torch.full((n, H), -1e30, device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, idx, score, reduce="amax", include_self=True)
    ex = (score - mx[index]).exp()
    den = torch.zeros((n, H), device=score.device, dtype=score.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


class RelAttnConv(nn.Module):
    """Relation-aware multi-head attention message passing with edge features."""

    def __init__(self, dim: int, n_rel: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim, self.heads, self.n_rel = dim, heads, n_rel
        self.W = nn.Parameter(torch.empty(n_rel, dim, dim))
        nn.init.xavier_uniform_(self.W.view(-1, dim))
        self.rel_emb = nn.Embedding(n_rel, dim)
        self.edge_lin = nn.Linear(edge_dim, dim)
        self.q = nn.Linear(dim, dim)
        self.self_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_type, edge_attr):
        n = h.shape[0]
        src, dst = edge_index
        dh = self.dim // self.heads
        msg = torch.zeros(src.shape[0], self.dim, device=h.device, dtype=h.dtype)
        for r in range(self.n_rel):  # per-relation projection, avoids an [E, d, d] gather
            m = edge_type == r
            if m.any():
                msg[m] = h[src[m]] @ self.W[r]
        msg = msg + self.edge_lin(edge_attr) + self.rel_emb(edge_type)
        q = self.q(h)[dst].view(-1, self.heads, dh)
        k = msg.view(-1, self.heads, dh)
        alpha = scatter_softmax((q * k).sum(-1) / math.sqrt(dh), dst, n)
        alpha_d = self.drop(alpha) if self.training else alpha
        agg = torch.zeros(n, self.heads, dh, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, alpha_d.unsqueeze(-1) * k)
        out = agg.view(n, self.dim) + self.self_lin(h)
        return h + self.drop(F.gelu(self.norm(out))), alpha.mean(1)


class TxHead(nn.Module):
    """Transaction-level risk: both endpoints' account representations + the transaction's own
    attributes (amount, payment rail, currency conversion, time of day, cross-bank flag)."""

    def __init__(self, rep_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * rep_dim + TX_DIM, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, rep, x_tx, src, dst):
        return self.net(torch.cat([rep[src], rep[dst], x_tx], 1)).squeeze(1)


class TxMixin:
    def tx_logits(self, view: GraphView, rep, sel=None, chunk: int = 65536):
        """Logits for the view's transactions (optionally a subset `sel` of view-local tx positions)."""
        if view.tx_src.numel() == 0:
            return torch.zeros(0, device=rep.device)
        src, dst, x = view.tx_src, view.tx_dst, view.x_tx
        if sel is not None:
            src, dst, x = src[sel], dst[sel], x[sel]
        if self.training or len(src) <= chunk:
            return self.tx_head(rep, x, src, dst)
        return torch.cat([self.tx_head(rep, x[i:i + chunk], src[i:i + chunk], dst[i:i + chunk])
                          for i in range(0, len(src), chunk)])


class NestedHGNN(TxMixin, nn.Module):
    def __init__(self, f_acc: int, f_dev: int, f_mer: int, edge_dim: int = EDGE_DIM, hidden: int = 64,
                 inner_layers: int = 2, outer_layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 nested: bool = True, max_dist: int = 3):
        super().__init__()
        self.hidden, self.nested = hidden, nested
        self.proj_acc = nn.Sequential(nn.Linear(f_acc, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.proj_dev = nn.Linear(f_dev, hidden)
        self.proj_mer = nn.Linear(f_mer, hidden)
        self.ghost_vec = nn.Parameter(torch.zeros(hidden))
        self.ex_lin = nn.Linear(hidden, hidden)        # maps exchanged cross-bank embeddings
        self.dist_emb = nn.Embedding(max_dist + 1, hidden)
        self.inner = nn.ModuleList([RelAttnConv(hidden, N_INNER_REL, edge_dim, heads, dropout) for _ in range(inner_layers)])
        self.gate = nn.Linear(hidden, 1)
        self.readout = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.outer = nn.ModuleList([RelAttnConv(hidden, N_OUTER_REL, edge_dim, heads, dropout) for _ in range(outer_layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.tx_head = TxHead(2 * hidden, hidden, dropout)

    # ---------------------------------------------------------------- helpers
    def ghost_init(self, view: GraphView, ghost_emb, ghost_mask):
        g = self.ghost_vec.expand(view.n_ghost, -1)
        if ghost_emb is not None and view.n_ghost:
            g = g + self.ex_lin(ghost_emb) * ghost_mask.unsqueeze(1).float()
        return g

    def node_init(self, view: GraphView, ghost_emb=None, ghost_mask=None):
        """Initial embeddings for every node of the view (type-specific encoders)."""
        return torch.cat([self.proj_acc(view.x_acc), self.ghost_init(view, ghost_emb, ghost_mask),
                          self.proj_dev(view.x_dev), self.proj_mer(view.x_mer)], 0)

    # ---------------------------------------------------------------- inner (nested) level
    def encode_subgraphs(self, view: GraphView, sub, ghost_emb=None, ghost_mask=None, h_view=None,
                         return_attention=False):
        if h_view is None:
            h_view = self.node_init(view, ghost_emb, ghost_mask)
        if isinstance(sub, (list, tuple)):  # chunked roots: bounded memory on large real graphs
            if len(sub) == 1:
                return self.encode_subgraphs(view, sub[0], h_view=h_view, return_attention=return_attention)
            zs = []
            for c in sub:
                if self.training and torch.is_grad_enabled():
                    zs.append(checkpoint(lambda hv, c=c: self.encode_subgraphs(view, c, h_view=hv)[0], h_view,
                                         use_reentrant=False))
                else:
                    zs.append(self.encode_subgraphs(view, c, h_view=h_view)[0])
            return torch.cat(zs, 0), []
        if not self.nested:
            return h_view[torch.as_tensor(sub.roots, device=h_view.device)], []
        h = h_view[sub.node_idx] + self.dist_emb(sub.dist.clamp(max=self.dist_emb.num_embeddings - 1))
        atts = []
        for conv in self.inner:
            h, a = conv(h, sub.edge_index, sub.edge_type, sub.edge_attr)
            atts.append(a)
        B = len(sub.roots)
        root = h[sub.root_pos]
        cnt = torch.zeros(B, device=h.device).index_add_(0, sub.batch, torch.ones_like(sub.batch, dtype=h.dtype)).clamp(min=1)
        mean = torch.zeros(B, self.hidden, device=h.device).index_add_(0, sub.batch, h) / cnt.unsqueeze(1)
        gate = scatter_softmax(self.gate(h), sub.batch, B)
        attn_pool = torch.zeros(B, self.hidden, device=h.device).index_add_(0, sub.batch, gate * h)
        z = self.readout(torch.cat([root, mean, attn_pool], 1))
        return z, (atts if return_attention else []) + ([gate.squeeze(1)] if return_attention else [])

    # ---------------------------------------------------------------- outer level
    def outer_forward(self, view: GraphView, z_own, ghost_emb=None, ghost_mask=None, return_attention=False):
        h = torch.cat([z_own, self.ghost_init(view, ghost_emb, ghost_mask)], 0)
        atts = []
        for conv in self.outer:
            h, a = conv(h, view.acc_edge_index, view.acc_edge_type, view.acc_edge_attr)
            atts.append(a)
        logits = self.head(torch.cat([h[: view.n_own], z_own], 1)).squeeze(1)
        return logits, h, atts

    @staticmethod
    def representation(view: GraphView, h_all, z_own):
        """[outer embedding | subgraph embedding] for every view account (ghosts have no z)."""
        z_pad = torch.cat([z_own, z_own.new_zeros(view.n_ghost, z_own.shape[1])], 0)
        return torch.cat([h_all, z_pad], 1)

    def forward(self, view: GraphView, sub: SubgraphBatch, ghost_emb=None, ghost_mask=None, return_attention=False):
        z, inner_att = self.encode_subgraphs(view, sub, ghost_emb, ghost_mask, return_attention=return_attention)
        logits, h, outer_att = self.outer_forward(view, z, ghost_emb, ghost_mask, return_attention)
        aux = {"z": z, "rep": self.representation(view, h, z)}
        if return_attention:
            aux.update({"h": h, "inner": inner_att, "outer": outer_att})
        return logits, aux


class FeatureMLP(TxMixin, nn.Module):
    """Feature-only baseline (no graph): what a conventional per-bank model sees. Its transaction
    head only knows the bank's own customer features; counterparties at other banks are blank."""

    def __init__(self, f_acc: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f_acc, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 1)
        self.tx_head = TxHead(hidden, hidden, dropout)

    def forward(self, view: GraphView, sub=None, ghost_emb=None, ghost_mask=None, return_attention=False):
        e = self.enc(view.x_acc)
        rep = torch.cat([e, e.new_zeros(view.n_ghost, e.shape[1])], 0)
        return self.out(e).squeeze(1), {"rep": rep}
