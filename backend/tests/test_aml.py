"""Real-data checks on the IBM AML consortium (skipped if the dataset cache is absent)."""
import numpy as np
import pytest

from app.aml import CACHE

pytestmark = pytest.mark.skipif(not any(CACHE.glob("HI-Small_*.npz")) if CACHE.exists() else True,
                                reason="IBM AML cache not built (run the server once with FRAUD_DATASET=ibm_aml)")


@pytest.fixture(scope="module")
def aml():
    from app.aml import load_ibm_aml
    return load_ibm_aml()


def test_cleaning_report(aml):
    r = aml.meta
    assert r["dropped_duplicates"] >= 0 and r["rates_usd"]["US Dollar"] == 1.0
    assert 0.5 < r["rates_usd"]["Euro"] < 2.0, "inferred FX rate out of range"
    assert aml.n_banks == len(r["members"]) and aml.is_member.sum() == r["n_member_accounts"]
    ts = aml.tx.view("ts")
    assert np.all(np.diff(ts) >= 0) and ts.max() < r["cutoff_day"] * 86400
    assert all(f[4] >= r["cutoff_day"] * 86400 for f in aml.future[:100])
    assert (aml.tx.view("src") != aml.tx.view("dst")).all(), "self-transfers must be removed from the graph"


def test_every_transaction_touches_a_member(aml):
    m = aml.is_member
    assert (m[aml.tx.view("src")] | m[aml.tx.view("dst")]).all()


def test_split_semantics(aml):
    from app.train import make_split
    s = make_split(aml)
    assert (s[~aml.is_member] == -1).all()
    assert (s[aml.live_pos & aml.is_member] == 3).all()
    for k in (0, 1, 2):
        assert aml.acc_label[s == k].sum() > 0
    # ring-disjoint: a historical ring's members never straddle train and test
    for r in aml.rings:
        ss = {int(s[a]) for a in r["members"] if s[a] in (0, 1, 2) and aml.acc_label[a]}
        assert not ({0, 2} <= ss)


def test_features_finite(aml):
    from app.features import account_features, feature_names
    ids = np.flatnonzero(aml.is_member)[:500]
    X = account_features(aml, ids)
    assert X.shape == (500, len(feature_names(aml))) and np.isfinite(X).all()
