"""Synthetic multi-bank transaction world with injected cross-bank fraud rings.

Real inter-bank datasets are not public, so we simulate a consortium of banks whose
customers transact with each other, use devices and pay merchants. Fraud rings
(layering chains, fan-in/fan-out mule networks, round-tripping cycles and device
farms) are spread deliberately across several banks so that no single bank sees the
whole ring. Legitimate "hard negatives" (businesses, payroll, households sharing a
device, people moving money between their own accounts) make the task non-trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DAY = 86400.0
T0_EPOCH = 1788220800.0  # 2026-09-01 00:00 UTC, used only for display
BANK_NAMES = ["Aurora Bank", "Bluewater Credit", "Cedar Trust", "Delta Savings", "Evergreen Bank", "Fjord Financial"]
PATTERNS = ["layering_chain", "fan_in_fan_out", "round_tripping", "device_farm"]

TX_TRANSFER = 0
TX_MERCHANT = 1


class TxStore:
    """Append-only columnar transaction store (amortised O(1) appends for streaming)."""

    cols = ("src", "dst", "merchant", "amount", "ts", "kind", "fmt", "xcur", "label")

    def __init__(self, capacity: int = 1024):
        self.n = 0
        self._cap = capacity
        self.src = np.zeros(capacity, np.int64)
        self.dst = np.full(capacity, -1, np.int64)
        self.merchant = np.full(capacity, -1, np.int64)
        self.amount = np.zeros(capacity, np.float64)
        self.ts = np.zeros(capacity, np.float64)
        self.kind = np.zeros(capacity, np.int8)
        self.fmt = np.full(capacity, -1, np.int8)    # payment format code (-1 = unknown)
        self.xcur = np.zeros(capacity, np.int8)      # 1 = currency conversion between legs
        self.label = np.zeros(capacity, np.int8)     # ground-truth laundering flag (evaluation only)

    def _grow(self, need: int):
        if need <= self._cap:
            return
        cap = max(need, self._cap * 2)
        for c in self.cols:
            old = getattr(self, c)
            new = np.full(cap, -1 if c in ("dst", "merchant", "fmt") else 0, old.dtype)
            new[: self.n] = old[: self.n]
            setattr(self, c, new)
        self._cap = cap

    def add(self, src, dst, merchant, amount, ts, kind, fmt=-1, xcur=0, label=0) -> int:
        self._grow(self.n + 1)
        i = self.n
        self.src[i], self.dst[i], self.merchant[i] = src, dst, merchant
        self.amount[i], self.ts[i], self.kind[i] = amount, ts, kind
        self.fmt[i], self.xcur[i], self.label[i] = fmt, xcur, label
        self.n += 1
        return i

    @classmethod
    def from_arrays(cls, **cols):
        n = len(cols["src"])
        t = cls(capacity=max(1024, int(n * 1.3)))
        for c, v in cols.items():
            getattr(t, c)[:n] = v
        t.n = n
        return t

    def extend(self, rows):
        for r in rows:
            self.add(*r)

    def view(self, c):
        return getattr(self, c)[: self.n]

    def sort_by_time(self):
        order = np.argsort(self.view("ts"), kind="stable")
        for c in self.cols:
            getattr(self, c)[: self.n] = self.view(c)[order]


@dataclass
class World:
    n_banks: int
    bank_names: list
    acc_bank: np.ndarray        # [N] bank index of each account
    acc_open_ts: np.ndarray     # [N] account opening time (seconds, relative to T0)
    acc_label: np.ndarray       # [N] 1 = fraud ring member (ground truth)
    acc_ring: np.ndarray        # [N] ring id or -1
    acc_role: list              # [N] role string
    acc_devices: list           # [N] list of device ids
    n_devices: int
    merchant_risk: np.ndarray   # [M] 1 = high-risk category (crypto / gift cards / remittance)
    merchant_names: list
    tx: TxStore
    rings: list                 # list of dicts: id, pattern, members, banks, live
    horizon: float              # end of simulated history (seconds)
    sleepers: list = field(default_factory=list)  # account pools reserved for live (streamed) rings
    rng_state: dict = field(default_factory=dict)
    source: str = "synthetic"                     # "synthetic" | "ibm_aml"
    t0_epoch: float = T0_EPOCH                    # epoch of ts == 0 (display only)
    acc_code: np.ndarray | None = None            # real institution code per account (cross-bank tests)
    acc_names: list | None = None                 # display ids (real data)
    future: list = field(default_factory=list)    # held-out real transactions replayed by the stream
    formats: list = field(default_factory=list)   # payment format names (real data)
    meta: dict = field(default_factory=dict)

    @property
    def is_member(self) -> np.ndarray:
        """Accounts held by consortium members (external institutions have acc_bank == n_banks)."""
        return self.acc_bank < self.n_banks

    def bank_name(self, b: int) -> str:
        return self.bank_names[b] if b < len(self.bank_names) else "External institution"

    @property
    def n_accounts(self):
        return len(self.acc_bank)

    @property
    def n_merchants(self):
        return len(self.merchant_risk)

    def account_name(self, a: int) -> str:
        if self.acc_names is not None:
            return self.acc_names[a]
        return f"{'ABCDEF'[self.acc_bank[a]]}-{a:05d}"

    def device_accounts(self):
        """device id -> list of accounts using it."""
        out: dict[int, list] = {}
        for a, devs in enumerate(self.acc_devices):
            for d in devs:
                out.setdefault(d, []).append(a)
        return out


def _daytime(rng, base_day: float, night_prob: float = 0.08) -> float:
    if rng.random() < night_prob:
        hour = rng.uniform(0, 6)
    else:
        hour = rng.normal(14, 3.5) % 24
    return base_day * DAY + hour * 3600 + rng.uniform(0, 3600) * 0.1


def generate_world(
    seed: int = 7,
    n_banks: int = 4,
    accounts_per_bank: int = 500,
    n_merchants: int = 120,
    n_rings: int = 40,
    days: int = 30,
    n_live_rings: int = 4,
) -> World:
    rng = np.random.default_rng(seed)
    N = n_banks * accounts_per_bank
    horizon = days * DAY
    acc_bank = np.repeat(np.arange(n_banks), accounts_per_bank)
    acc_open_ts = -rng.uniform(60, 3000, N) * DAY
    young = rng.random(N) < 0.12  # legit new customers
    acc_open_ts[young] = -rng.uniform(5, 90, young.sum()) * DAY
    acc_label = np.zeros(N, np.int64)
    acc_ring = np.full(N, -1, np.int64)
    acc_role = ["customer"] * N

    # ---- devices: everyone has a phone; some have a second device; households share one
    acc_devices = [[i] for i in range(N)]
    n_dev = N
    for a in rng.choice(N, int(0.25 * N), replace=False):
        acc_devices[a].append(n_dev)
        n_dev += 1
    for _ in range(int(0.08 * N)):  # households: 2-3 accounts (possibly at different banks) share a tablet
        members = rng.choice(N, rng.integers(2, 4), replace=False)
        for m in members:
            acc_devices[m].append(n_dev)
        n_dev += 1

    # ---- merchants
    merchant_risk = (rng.random(n_merchants) < 0.1).astype(np.int64)
    merchant_risk[:3] = 1
    risky_names = ["CryptoXchange", "GiftCardHub", "QuickRemit", "CoinVault", "PrepaidPlus"]
    normal_names = ["GroceryMart", "FuelStop", "CityPharmacy", "StreamFlix", "ElectroWorld", "CafeBrew",
                    "BookNook", "MetroTransit", "HomeDepotX", "FashionLane", "PetPals", "UtilityCo"]
    merchant_names = []
    for m in range(n_merchants):
        pool = risky_names if merchant_risk[m] else normal_names
        merchant_names.append(f"{pool[m % len(pool)]} #{m}")
    risky_merchants = np.flatnonzero(merchant_risk)
    merchant_pop = 1.0 / np.arange(1, n_merchants + 1) ** 0.8
    merchant_pop[risky_merchants] *= 0.15
    merchant_pop /= merchant_pop.sum()

    tx = TxStore(capacity=1 << 16)

    # ---- choose ring members (history rings + sleeper pools for live rings)
    free = rng.permutation(N).tolist()
    roles = {}

    def take(k, banks):
        """Take k free accounts spread over the given banks (round robin)."""
        out = []
        bi = 0
        guard = 0
        while len(out) < k and guard < 100000:
            guard += 1
            b = banks[bi % len(banks)]
            for j, a in enumerate(free):
                if acc_bank[a] == b:
                    out.append(free.pop(j))
                    break
            bi += 1
        return out

    rings = []
    for r in range(n_rings):
        pattern = PATTERNS[r % len(PATTERNS)]
        nb = int(rng.integers(2, min(n_banks, 4) + 1))
        banks = rng.choice(n_banks, nb, replace=False).tolist()
        size = int(rng.integers(5, 13))
        members = take(size, banks)
        rings.append({"id": r, "pattern": pattern, "members": members,
                      "banks": sorted({int(acc_bank[m]) for m in members}), "live": False})
    sleepers = []
    for r in range(n_live_rings):
        banks = rng.choice(n_banks, min(n_banks, 3), replace=False).tolist()
        sleepers.append(take(int(rng.integers(6, 10)), banks))

    # ---- business accounts & payroll (legit hard negatives)
    businesses = [free.pop() for _ in range(int(0.03 * N))]
    employers = [free.pop() for _ in range(int(0.01 * N))]
    for b in businesses:
        acc_role[b] = "business"
    for e in employers:
        acc_role[e] = "employer"
    dual = [(free.pop(), free.pop()) for _ in range(int(0.05 * N))]  # same person, two banks

    # ---- normal behaviour for every account
    contacts = []
    for a in range(N):
        k = int(rng.integers(2, 9))
        same = np.flatnonzero(acc_bank == acc_bank[a])
        cs = []
        for _ in range(k):
            pool = same if rng.random() < 0.65 else np.arange(N)
            c = int(rng.choice(pool))
            if c != a:
                cs.append(c)
        if rng.random() < 0.3 and businesses:
            cs.append(int(rng.choice(businesses)))
        contacts.append(cs)

    for a in range(N):
        n_t = rng.poisson(6)
        for _ in range(n_t):
            if not contacts[a]:
                break
            d = contacts[a][rng.integers(len(contacts[a]))]
            amt = float(np.round(rng.lognormal(5.3, 1.1), 2))
            if rng.random() < 0.15:
                amt = float(np.round(amt, -1))
            tx.add(a, d, -1, amt, _daytime(rng, rng.uniform(0, days)), TX_TRANSFER)
        for _ in range(rng.poisson(14)):
            m = int(rng.choice(n_merchants, p=merchant_pop))
            amt = float(np.round(rng.lognormal(3.6, 0.9), 2))
            tx.add(a, -1, m, amt, _daytime(rng, rng.uniform(0, days)), TX_MERCHANT)

    for a in rng.choice(N, int(0.05 * N), replace=False):  # legit large payments near the 10k threshold
        for _ in range(rng.integers(1, 3)):
            d = int(rng.choice(contacts[a] or [int(rng.integers(N))]))
            tx.add(int(a), d, -1, float(np.round(rng.uniform(8500, 9999), 2)), _daytime(rng, rng.uniform(0, days)), TX_TRANSFER)
    for e in employers:  # payroll fan-out on the 1st and 15th
        staff = rng.choice(N, int(rng.integers(15, 35)), replace=False)
        for day in (0, 14, 28):
            for s in staff:
                if s != e:
                    tx.add(e, int(s), -1, float(np.round(rng.normal(3200, 600), 2)), day * DAY + 9 * 3600 + rng.uniform(0, 600), TX_TRANSFER)
    for b in businesses:  # businesses forward takings to suppliers (fan-in then fan-out)
        sup = rng.choice(N, 5, replace=False)
        for _ in range(rng.integers(4, 9)):
            tx.add(b, int(rng.choice(sup)), -1, float(np.round(rng.lognormal(7.5, 0.6), 2)), _daytime(rng, rng.uniform(0, days)), TX_TRANSFER)
    for a, b in dual:  # legit pass-through between own accounts at two banks
        for _ in range(rng.integers(2, 6)):
            t = _daytime(rng, rng.uniform(0, days - 1))
            amt = float(np.round(rng.lognormal(7, 0.5), 2))
            tx.add(a, b, -1, amt, t, TX_TRANSFER)
            if rng.random() < 0.5:
                tx.add(b, int(rng.choice(contacts[b] or [a])), -1, amt * 0.9, t + rng.uniform(600, 7200), TX_TRANSFER)
        acc_devices[b].append(acc_devices[a][0])  # same phone for both accounts

    # ---- fraud rings
    n_dev_box = [n_dev]

    def ring_devices(members, p_share):
        farm = list(range(n_dev_box[0], n_dev_box[0] + int(rng.integers(1, 4))))
        n_dev_box[0] += len(farm)
        for m in members:
            if rng.random() < p_share:
                acc_devices[m].append(int(rng.choice(farm)))

    def inject(ring, t_start, t_end, live=False):
        rows = []
        pat, mem = ring["pattern"], list(ring["members"])
        bursts = int(rng.integers(2, 5)) if not live else 1
        legit_pool = np.setdiff1d(np.arange(N), np.concatenate([np.array(r["members"]) for r in rings] + [np.array(s) for s in sleepers]))
        for _ in range(bursts):
            t = rng.uniform(t_start, max(t_start + 1, t_end - 2 * DAY)) if not live else t_start
            structuring = rng.random() < 0.25
            k = max(3, int(round(len(ring['members']) * rng.uniform(0.6, 1.0))))
            mem = list(rng.permutation(ring['members'])[:k]) if not live else list(ring['members'])

            def amount(base=None):
                if structuring:
                    return float(np.round(rng.uniform(9000, 9950), 2))
                return float(np.round(base if base else rng.lognormal(7.2, 0.7), 2))

            victims = rng.choice(legit_pool, int(rng.integers(2, 7)), replace=False)
            if pat == "layering_chain":
                layers = np.array_split(np.array(mem), min(len(mem), int(rng.integers(3, 5))))
                for v in victims:
                    rows.append((int(v), int(rng.choice(layers[0])), -1, amount(), t, TX_TRANSFER))
                    t += rng.uniform(60, 1800)
                for li in range(len(layers) - 1):
                    for s in layers[li]:
                        for _ in range(int(rng.integers(1, 3))):
                            t += rng.uniform(120, 5400)
                            rows.append((int(s), int(rng.choice(layers[li + 1])), -1, amount() * rng.uniform(0.95, 0.99), t, TX_TRANSFER))
                for s in layers[-1]:
                    t += rng.uniform(300, 3600)
                    rows.append((int(s), -1, int(rng.choice(risky_merchants)), amount() * 0.97, t, TX_MERCHANT))
            elif pat == "fan_in_fan_out":
                collectors, rest = mem[:2], mem[2:]
                exit_ = rest[-1]
                splitters = rest[:-1] or [exit_]
                for v in victims:
                    rows.append((int(v), int(rng.choice(collectors)), -1, amount(rng.lognormal(7.5, 0.5)), t, TX_TRANSFER))
                    t += rng.uniform(60, 900)
                for c in collectors:
                    for s in splitters:
                        t += rng.uniform(60, 1200)
                        rows.append((int(c), int(s), -1, amount(rng.lognormal(7.0, 0.3)), t, TX_TRANSFER))
                for s in splitters:
                    t += rng.uniform(300, 3600)
                    rows.append((int(s), int(exit_), -1, amount(rng.lognormal(6.9, 0.3)), t, TX_TRANSFER))
                rows.append((int(exit_), -1, int(rng.choice(risky_merchants)), amount(rng.lognormal(8.5, 0.3)), t + 1800, TX_MERCHANT))
            elif pat == "round_tripping":
                base = rng.lognormal(8.0, 0.5)
                for _loop in range(int(rng.integers(2, 4))):
                    order = rng.permutation(mem)
                    amt = base
                    for i in range(len(order)):
                        t += rng.uniform(600, 7200)
                        amt *= rng.uniform(0.97, 1.0)
                        rows.append((int(order[i]), int(order[(i + 1) % len(order)]), -1, amount(amt), t, TX_TRANSFER))
                rows.append((int(victims[0]), int(mem[0]), -1, amount(base), t - 3600, TX_TRANSFER))
            else:  # device_farm: synthetic identities receive small deposits, forward to an exit
                exit_ = mem[-1]
                for m in mem[:-1]:
                    for _ in range(int(rng.integers(2, 4))):
                        v = int(rng.choice(victims))
                        rows.append((v, int(m), -1, float(np.round(rng.uniform(200, 900), 2)), t, TX_TRANSFER))
                        t += rng.uniform(60, 600)
                        rows.append((int(m), int(exit_), -1, float(np.round(rng.uniform(180, 880), 2)), t + rng.uniform(60, 900), TX_TRANSFER))
                rows.append((int(exit_), -1, int(rng.choice(risky_merchants)), float(np.round(rng.uniform(3000, 9000), 2)), t + 3600, TX_MERCHANT))
        # camouflage: members also behave a bit normally
        for m in ring['members']:
            for _ in range(rng.poisson(4 if not live else 0)):
                rows.append((int(m), -1, int(rng.choice(n_merchants, p=merchant_pop)), float(np.round(rng.lognormal(3.6, 0.9), 2)), _daytime(rng, rng.uniform(t_start / DAY, max(t_start / DAY + 0.1, t_end / DAY))), TX_MERCHANT))
        mem_set = set(ring['members'])
        # transaction-level ground truth: ring transfers and cash-outs to high-risk merchants
        return [r + (-1, 0, int((r[5] == TX_TRANSFER and (r[0] in mem_set or r[1] in mem_set))
                                 or (r[5] == TX_MERCHANT and merchant_risk[r[2]] == 1 and r[0] in mem_set)))
                for r in rows]

    for ring in rings:
        mem = ring["members"]
        recent = rng.random(len(mem)) < 0.35
        for i, m in enumerate(mem):
            acc_label[m] = 1
            acc_ring[m] = ring["id"]
            acc_role[m] = "mule"
            if recent[i]:
                acc_open_ts[m] = -rng.uniform(5, 90) * DAY
        ring_devices(mem, 0.85 if ring["pattern"] == "device_farm" else 0.2)
        tx.extend(inject(ring, 0.0, horizon))

    tx.sort_by_time()
    world = World(
        n_banks=n_banks, bank_names=BANK_NAMES[:n_banks], acc_bank=acc_bank, acc_open_ts=acc_open_ts,
        acc_label=acc_label, acc_ring=acc_ring, acc_role=acc_role, acc_devices=acc_devices, n_devices=n_dev_box[0],
        merchant_risk=merchant_risk, merchant_names=merchant_names, tx=tx, rings=rings, horizon=horizon,
        sleepers=sleepers,
    )
    world._inject = inject  # keep generator closure for live ring injection
    world._ring_devices = ring_devices
    world._rng = rng
    world._contacts = contacts
    world._merchant_pop = merchant_pop
    return world


def activate_live_ring(world: World, t_now: float):
    """Turn a pool of sleeper accounts into a new fraud ring and return its transactions."""
    if not world.sleepers:
        return None, []
    rng = world._rng
    members = world.sleepers.pop(0)
    rid = len(world.rings)
    pattern = PATTERNS[rid % len(PATTERNS)]
    ring = {"id": rid, "pattern": pattern, "members": members,
            "banks": sorted({int(world.acc_bank[m]) for m in members}), "live": True, "activated_ts": t_now}
    world.rings.append(ring)
    for m in members:
        world.acc_label[m] = 1
        world.acc_ring[m] = rid
        world.acc_role[m] = "mule"
    world._ring_devices(members, 0.9 if pattern == "device_farm" else 0.35)
    world.n_devices = max(world.n_devices, max(d for m in members for d in world.acc_devices[m]) + 1)
    rows = world._inject(ring, t_now, t_now + DAY, live=True)
    rows.sort(key=lambda r: r[4])
    return ring, rows


def normal_traffic(world: World, t_now: float, n: int):
    """Generate n legitimate transactions around time t_now."""
    rng = world._rng
    N = world.n_accounts
    rows = []
    for _ in range(n):
        a = int(rng.integers(N))
        t = t_now + rng.uniform(0, 1)
        if rng.random() < 0.45 and world._contacts[a]:
            d = world._contacts[a][rng.integers(len(world._contacts[a]))]
            rows.append((a, int(d), -1, float(np.round(rng.lognormal(5.3, 1.1), 2)), t, TX_TRANSFER))
        else:
            m = int(rng.choice(world.n_merchants, p=world._merchant_pop))
            rows.append((a, -1, m, float(np.round(rng.lognormal(3.6, 0.9), 2)), t, TX_MERCHANT))
    return rows
