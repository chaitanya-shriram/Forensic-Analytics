import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FRAUD_NO_AUTOSTART", "1")

SMALL = dict(n_banks=3, accounts_per_bank=120, n_merchants=40, n_rings=12, n_live_rings=2)


@pytest.fixture(scope="session")
def world():
    from app.data import generate_world
    return generate_world(seed=3, **SMALL)


@pytest.fixture(scope="session")
def prep(world):
    from app.train import prepare
    return prepare(world, seed=0, device="cpu")
