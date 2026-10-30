import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import SMALL


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import main
    e = main.engine
    e.dataset = "synthetic"
    e.world_kwargs = dict(SMALL)
    e.train_cfg = {"epochs": 20, "rounds": 3, "local_epochs": 2}
    e.artifacts = tmp_path_factory.mktemp("artifacts")
    e.device = "cpu"
    e.seed = 3
    with TestClient(main.app) as c:
        e.start()
        yield c
    e.stop_stream()


def test_status_and_overview(client):
    s = client.get("/api/status").json()
    assert s["ready"] is True
    o = client.get("/api/overview").json()
    assert o["accounts"] == SMALL["n_banks"] * SMALL["accounts_per_bank"]
    assert len(o["banks"]) == SMALL["n_banks"]
    assert sum(h["count"] for h in o["score_hist"]) == o["accounts"]
    assert o["daily"]


def test_metrics_lists_all_models(client):
    m = client.get("/api/metrics").json()
    keys = {x["key"] for x in m["models"]}
    assert keys == {"mlp", "flat_gnn", "nested_central", "nested_local", "nested_federated"}
    assert all(0 <= x["auc"] <= 1 for x in m["models"])


def test_accounts_and_explanation(client):
    lst = client.get("/api/accounts", params={"limit": 5}).json()
    scores = [a["score"] for a in lst["items"]]
    assert scores == sorted(scores, reverse=True)
    a = lst["items"][0]["id"]
    d = client.get(f"/api/accounts/{a}").json()
    ex = d["explanation"]
    assert ex["features"] and ex["outer_attention"] is not None and ex["subgraph_pooling"]
    assert set(d["model_scores"]) >= {"nested_federated", "mlp"}
    assert any(n["id"] == f"a{a}" for n in d["graph"]["nodes"])
    assert client.get("/api/accounts/999999").status_code == 404


def test_rings(client):
    rings = client.get("/api/rings").json()
    assert isinstance(rings, list)
    if rings:
        r = client.get(f"/api/rings/{rings[0]['id']}").json()
        assert len(r["members"]) == r["size"]
        assert r["graph"]["nodes"]


def test_submit_transaction_rescoring(client):
    r = client.post("/api/transactions", json={"src": 1, "dst": 200, "amount": 9500}).json()
    assert r["latency"]["total_ms"] > 0 and r["tx"][0]["amount"] == 9500
    assert client.post("/api/transactions", json={"src": 1, "amount": 10}).status_code == 400
    assert client.post("/api/transactions", json={"src": 1, "merchant": 2, "amount": 10}).status_code == 200


def test_stream_and_ring_injection(client):
    from app import main
    inj = client.post("/api/stream/inject-ring").json()
    assert inj["size"] >= 3 and inj["pending_tx"] > 0
    with client.websocket_connect("/ws") as ws:
        client.post("/api/stream/start", json={"speed": 3600, "rate": 3})
        types = set()
        for _ in range(20):
            ev = ws.receive_json()
            types.add(ev["type"])
            if "batch" in types:
                break
        client.post("/api/stream/stop")
    assert "batch" in types
    st = client.get("/api/stream").json()
    assert st["n_tx"] > 0 and st["latencies"]
    assert main.engine.live_rings


def test_experiment_validation(client):
    assert client.post("/api/experiments", json={"rounds": 0}).status_code == 422
    p = client.get("/api/privacy").json()
    assert p["secure_aggregation"] is True and p["never_shared"]


def test_review_queue_and_thresholds(client):
    o = client.get("/api/overview").json()
    thr = o["thresholds"]
    assert 0 < thr["tx"] <= 1 and 0 < thr["account"] <= 1
    q = client.get("/api/review", params={"limit": 10}).json()
    risks = [t["risk"] for t in q["items"]]
    assert risks == sorted(risks, reverse=True)
    assert all(r >= thr["tx"] for r in risks)
    assert q["total_flagged"] >= len(q["items"])


def test_nested_walkthrough(client):
    a = client.get("/api/nested/sample").json()["account"]
    d = client.get(f"/api/nested/{a}").json()
    ex = d["explanation"]
    sg = ex["subgraph"]
    assert sg["nodes"][0]["hop"] == 0, "root must be the first subgraph node"
    assert all(0 <= e["source"] < len(sg["nodes"]) and 0 <= e["target"] < len(sg["nodes"]) for e in sg["edges"])
    assert len(ex["z"]) == 64 and 0 <= ex["p_acc"] <= 1
    assert d["transactions"] and d["thresholds"]["combine"] in ("acc", "txmax", "mean")
    assert client.get("/api/nested/999999").status_code == 404
