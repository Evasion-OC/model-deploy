"""API tests against the exported artifacts. The refiner artifact is exported on the fly if missing
(needs torch); GPT-2 tests are skipped when its artifact is absent (CI exports it first)."""
import os, subprocess, sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
os.environ["ARTIFACTS"] = ART
sys.path.insert(0, ROOT)
from serve.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def refiner_artifact():
    if not os.path.exists(os.path.join(ART, "refiner_encoder.onnx")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "export", "export_refiner.py"), "--skip-eval"], cwd=ROOT)


def probe(n=300, d=8, in_dim=22, seed=0):
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((n, d)))
    src = np.arange(n); dst = (src + 1) % n
    adj = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    w = np.ones(adj.shape[1], dtype=np.float32)
    deg = np.bincount(adj[1], minlength=n).astype(np.float32)
    return dict(x=rng.standard_normal((n, in_dim)).astype(np.float32).tolist(), eigvals=np.sort(rng.random(d)).tolist(),
                eigvecs=V.astype(np.float32).tolist(), adj_indices=adj.tolist(), adj_weights=w.tolist(), deg_inv=(1 / deg).tolist())


def test_health():
    r = client.get("/health"); assert r.status_code == 200
    assert r.json()["status"] == "ok" and "refiner_encoder.onnx" in r.json()["artifacts"]


def test_refiner_embed_shapes_and_determinism():
    p = probe()
    r1 = client.post("/refiner/embed", json=p); r2 = client.post("/refiner/embed", json=p)
    assert r1.status_code == 200, r1.text
    b = r1.json(); assert b["n"] == 300 and b["d_model"] > 0 and len(b["graph_embedding"]) == b["d_model"]
    assert np.allclose(b["graph_embedding"], r2.json()["graph_embedding"])


def test_refiner_permutation_invariance_of_graph_embedding():
    p = probe(); n = len(p["x"]); rng = np.random.default_rng(1); perm = rng.permutation(n); inv = np.argsort(perm)
    q = dict(p); q["x"] = np.asarray(p["x"])[perm].tolist(); q["eigvecs"] = np.asarray(p["eigvecs"])[perm].tolist()
    q["adj_indices"] = inv[np.asarray(p["adj_indices"])].tolist(); q["deg_inv"] = np.asarray(p["deg_inv"])[perm].tolist()
    a = client.post("/refiner/embed", json=p).json()["graph_embedding"]; b = client.post("/refiner/embed", json=q).json()["graph_embedding"]
    assert np.allclose(a, b, atol=1e-4)


def test_refiner_rejects_bad_shapes():
    p = probe(); p["eigvecs"] = p["eigvecs"][:10]
    assert client.post("/refiner/embed", json=p).status_code == 422


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "gpt2.onnx")) and not os.path.exists(os.path.join(ART, "gpt2_int8.onnx")),
                    reason="gpt2 artifact not exported")
def test_gpt2_score_and_generate():
    r = client.post("/gpt2/score", json={"text": "The quick brown fox jumps over the lazy dog."})
    assert r.status_code == 200 and 0 < r.json()["mean_nll"] < 20
    g = client.post("/gpt2/generate", json={"prompt": "The capital of France is", "max_new_tokens": 4})
    assert g.status_code == 200 and g.json()["text"].startswith("The capital of France is")
