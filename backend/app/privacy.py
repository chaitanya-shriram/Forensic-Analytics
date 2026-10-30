"""Privacy mechanisms used by the federated consortium.

* Keyed pseudonymisation (HMAC-SHA256) of account / device identifiers, so the coordinator
  can join records across banks without learning raw identifiers.
* PSI-style device registry: banks upload hashed device ids, the registry only returns
  aggregate counts (how many accounts / banks share a device).
* Secure aggregation: pairwise additive masks that cancel in the sum, so the server only
  learns the aggregate model update, never an individual bank's update.
* Differential privacy: L2 clipping + Gaussian noise for model updates and for exchanged
  cross-bank embeddings, with a simple RDP accountant.
"""
from __future__ import annotations

import hashlib
import hmac
import math

import numpy as np
import torch

CONSORTIUM_KEY = b"consortium-demo-key-rotate-in-production"


def pseudonym(kind: str, ident: int, key: bytes = CONSORTIUM_KEY) -> str:
    return hmac.new(key, f"{kind}:{ident}".encode(), hashlib.sha256).hexdigest()[:16]


def psi_device_counts(acc_devices, acc_bank, n_devices):
    """Each bank submits {hash(device)} for its customers; the registry returns per-device
    (n_accounts, n_banks). Returned indexed by device id for convenience of the bank side."""
    registry: dict[str, list] = {}
    for a, devs in enumerate(acc_devices):
        for d in devs:
            registry.setdefault(pseudonym("dev", d), []).append(int(acc_bank[a]))
    n_acc = np.zeros(n_devices)
    n_banks = np.zeros(n_devices)
    for d in range(n_devices):
        e = registry.get(pseudonym("dev", d), [])
        n_acc[d], n_banks[d] = len(e), len(set(e))
    return n_acc, n_banks


def flat(state: dict) -> torch.Tensor:
    return torch.cat([v.detach().float().reshape(-1) for v in state.values()])


def unflat(vec: torch.Tensor, like: dict) -> dict:
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[i:i + n].view_as(v).to(v.dtype)
        i += n
    return out


def clip(vec: torch.Tensor, max_norm: float):
    norm = vec.norm().item()
    return vec * min(1.0, max_norm / (norm + 1e-12)), norm


def secure_aggregate(updates: list, round_seed: int) -> torch.Tensor:
    """Sum client vectors with pairwise masks. Client i adds m_ij for j>i and subtracts m_ji
    for j<i; masks are derived from a pairwise shared seed (stand-in for a DH key agreement)."""
    n = len(updates)
    masked = []
    for i in range(n):
        v = updates[i].clone()
        for j in range(n):
            if i == j:
                continue
            a, b = min(i, j), max(i, j)
            g = torch.Generator().manual_seed(hash((round_seed, a, b)) & 0x7FFFFFFF)
            m = torch.randn(v.shape, generator=g) * 10.0
            v += m if i == a else -m
        masked.append(v)  # this is all the server receives from client i
    return torch.stack(masked).sum(0), masked


def gaussian_noise(shape, std, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(shape, generator=g) * std


def rdp_epsilon(noise_multiplier: float, steps: int, delta: float = 1e-5) -> float:
    """Gaussian mechanism composed `steps` times (no subsampling amplification): conservative."""
    if noise_multiplier <= 0:
        return math.inf
    best = math.inf
    for alpha in [1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256]:
        eps = steps * alpha / (2 * noise_multiplier ** 2) + math.log(1 / delta) / (alpha - 1)
        best = min(best, eps)
    return best
