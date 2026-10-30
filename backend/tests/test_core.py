import numpy as np
import torch

from app.data import TX_TRANSFER, generate_world
from app.features import account_features, feature_names
from app.graph import NODE_MERCHANT, extract_subgraphs
from app.model import NestedHGNN, scatter_softmax
from app.privacy import pseudonym, rdp_epsilon, secure_aggregate
from app.rings import detect_rings, ring_recall
from app.train import evaluate, exchange_embeddings, new_model, predict, train_central, train_federated

from conftest import SMALL


def test_world_is_deterministic_and_rings_cross_banks(world):
    w2 = generate_world(seed=3, **SMALL)
    assert w2.tx.n == world.tx.n
    assert np.allclose(w2.tx.view("amount"), world.tx.view("amount"))
    assert world.acc_label.sum() == sum(len(r["members"]) for r in world.rings)
    assert np.mean([len(r["banks"]) > 1 for r in world.rings]) > 0.8
    assert np.all(np.diff(world.tx.view("ts")) >= 0)


def test_feature_subset_matches_full(world):
    full = account_features(world)
    ids = [0, 5, 77, 200]
    assert full.shape == (world.n_accounts, len(feature_names(world)))
    assert np.allclose(account_features(world, ids), full[ids], atol=1e-5)
    assert np.isfinite(full).all()


def test_bank_view_partitions_accounts(prep, world):
    seen = set()
    for b, v in enumerate(prep.banks):
        own = v.acc_ids[: v.n_own]
        assert (world.acc_bank[own] == b).all()
        assert (world.acc_bank[v.acc_ids[v.n_own:]] != b).all(), "ghosts must belong to other banks"
        seen |= set(own.tolist())
        assert v.x_acc.shape[0] == v.n_own
    assert seen == set(range(world.n_accounts))


def test_subgraphs_are_rooted_and_induced(prep):
    v = prep.banks[0]
    sub = extract_subgraphs(v, [0, 1, 2])
    assert (sub.dist[sub.root_pos] == 0).all()
    assert (sub.node_idx[sub.root_pos].numpy() == [0, 1, 2]).all()
    b = sub.batch
    assert (b[sub.edge_index[0]] == b[sub.edge_index[1]]).all(), "edges must stay inside one subgraph"
    nt = v.node_type()[sub.node_idx.numpy()]
    # merchants are leaves: every hop-2 node is reached through a non-merchant hop-1 node
    dist = sub.dist.numpy()
    src, dst = sub.edge_index.numpy()
    for n in np.flatnonzero(dist == 2):
        parents = src[(dst == n) & (dist[src] == 1)]
        assert (nt[parents] != NODE_MERCHANT).any()


def test_scatter_softmax_normalises():
    score = torch.randn(10, 2)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3])
    a = scatter_softmax(score, idx, 4)
    s = torch.zeros(4, 2).index_add_(0, idx, a)
    assert torch.allclose(s, torch.ones(4, 2), atol=1e-6)


def test_nested_model_forward_backward(prep):
    m = new_model("nested", prep)
    v, s = prep.banks[0], prep.bank_subs[0]
    logits, aux = m(v, s, return_attention=True)
    assert logits.shape == (v.n_own,)
    assert len(aux["inner"]) == 3 and len(aux["outer"]) == 2
    logits.sum().backward()
    assert m.inner[0].W.grad is not None and m.outer[0].W.grad is not None
    flat = new_model("flat_gnn", prep)
    assert isinstance(flat, NestedHGNN) and not flat.nested
    assert flat(v, s)[0].shape == (v.n_own,)


def test_secure_aggregation_masks_cancel():
    ups = [torch.randn(50) for _ in range(4)]
    total, masked = secure_aggregate(ups, 11)
    assert torch.allclose(total, torch.stack(ups).sum(0), atol=1e-4)
    assert not torch.allclose(masked[0], ups[0])


def test_privacy_helpers():
    assert pseudonym("acc", 1) == pseudonym("acc", 1) != pseudonym("acc", 2)
    assert rdp_epsilon(2.0, 10) < rdp_epsilon(0.5, 10)


def test_ring_detection_on_oracle_scores(world):
    scores = world.acc_label.astype(float)
    rings = detect_rings(world, scores, 0.5)
    assert ring_recall(world, rings) >= 0.8
    assert np.mean([r["precision"] for r in rings]) == 1.0


def test_training_beats_chance(prep):
    m, hist = train_central("nested_central", prep, epochs=30)
    res = evaluate(prep, predict(m, prep, "nested_central"))
    assert res["auc"] > 0.8, res
    assert hist and "val_ap" in hist[-1]


def test_federated_training_with_exchange_and_dp(prep):
    m, hist, priv = train_federated(prep, rounds=3, local_epochs=2, exchange=True, dp_noise=0.1)
    assert len(hist) == 3 and priv["secure_aggregation"] and priv["epsilon_updates"] > 0
    pub, has = exchange_embeddings(m, prep)
    assert has.any()
    # only accounts that appear as ghosts at another bank are published
    needed = set()
    for v in prep.banks:
        needed |= set(v.acc_ids[v.n_own:].tolist())
    assert set(torch.nonzero(has).flatten().tolist()) == needed
    pred = predict(m, prep, "nested_federated", (pub, has))
    sc = pred["acc"]
    assert sc.shape == (prep.world.n_accounts,) and ((sc >= 0) & (sc <= 1)).all()
    seen = ~np.isnan(pred["tx"])
    transfers = prep.world.tx.view("kind") == 0
    assert seen[transfers].all() and not seen[~transfers].any()
    assert seen[transfers].mean() > 0.9, "every member-touching transfer should get a transaction risk"
    assert ((pred["tx"][seen] >= 0) & (pred["tx"][seen] <= 1)).all()


def test_transaction_split_never_trains_on_test_accounts(prep):
    t = prep.world.tx
    tr = prep.tx_split == 0
    for arr in (t.view("src"), t.view("dst")):
        ok = arr >= 0
        assert not np.any(prep.split[arr[tr & ok]] == 2)


def test_transaction_rings_on_oracle_risk(world):
    from app.rings import detect_tx_rings
    risk = world.tx.view("label").astype(float)
    rings = detect_tx_rings(world, risk, 0.5)
    assert rings and all(r["precision"] == 1.0 for r in rings)
    covered = set().union(*(set(r["members"]) for r in rings))
    ring_members = {m for r in world.rings for m in r["members"]}
    assert len(covered & ring_members) / len(ring_members) > 0.8
