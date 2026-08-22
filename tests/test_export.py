"""Export parity: the ONNX graph must reproduce the PyTorch forward, on graphs the export never saw."""
import os, subprocess, sys
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "export"))
from export_refiner import build_encoder, probe_graph, feeds_of, rel_drift  # noqa: E402
import onnxruntime as ort  # noqa: E402

ART = os.path.join(ROOT, "artifacts")


@pytest.fixture(scope="module")
def refiner():
    if not os.path.exists(os.path.join(ART, "refiner_encoder.onnx")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "export", "export_refiner.py"), "--skip-eval"], cwd=ROOT)
    enc, sd = build_encoder(os.path.join(ROOT, "models", "refiner", "spectral_refiner_k16_eps_0_03.pt"))
    return enc, sd, ort.InferenceSession(os.path.join(ART, "refiner_encoder.onnx"), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("n,seed", [(128, 11), (700, 12), (1500, 13)])
def test_refiner_onnx_matches_torch_on_unseen_graphs(refiner, n, seed):
    enc, sd, s = refiner
    t = probe_graph(n, sd["in_dim"], sd["n_eigs"], seed)
    with torch.no_grad():
        ref = enc(*t).numpy()
    out = s.run(["embeddings"], feeds_of(t))[0]
    assert out.shape == ref.shape
    assert rel_drift(out, ref) < 1e-5


def test_refiner_duplicate_destination_edges_are_summed(refiner):
    """The regression the index_add_ -> scatter_add change fixed: multi-edges must add, not overwrite."""
    enc, sd, s = refiner
    t = list(probe_graph(200, sd["in_dim"], sd["n_eigs"], 21))
    ai = t[3]; t[3] = torch.cat([ai, ai], dim=1); t[4] = torch.cat([t[4], t[4]])  # duplicate every edge
    with torch.no_grad():
        ref = enc(*t).numpy()
    assert rel_drift(s.run(["embeddings"], feeds_of(t))[0], ref) < 1e-5


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "gpt2.onnx")), reason="gpt2 artifact not exported")
def test_gpt2_onnx_matches_torch():
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    s = ort.InferenceSession(os.path.join(ART, "gpt2.onnx"), providers=["CPUExecutionProvider"])
    for seq in (37, 300):  # the export is batch-1 with a dynamic sequence axis; exercise two lengths
        ids = torch.randint(0, 50257, (1, seq))
        with torch.no_grad():
            ref = m(input_ids=ids, use_cache=False).logits.numpy()
        out = s.run(["logits"], {"input_ids": ids.numpy()})[0]
        assert out.shape == ref.shape and np.abs(out - ref).max() < 1e-2
