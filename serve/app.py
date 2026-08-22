"""FastAPI front for the exported ONNX models. Inference runs in ONNX Runtime only: the container
has no PyTorch. Artifacts are read from $ARTIFACTS (default ./artifacts).

    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ART = os.environ.get("ARTIFACTS", "artifacts")
app = FastAPI(title="model-deploy", description="ONNX Runtime serving for the partition refiner encoder and GPT-2")
_sessions: dict[str, ort.InferenceSession] = {}


def session(name: str) -> ort.InferenceSession:
    if name not in _sessions:
        path = os.path.join(ART, name)
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"artifact {name} not present in {ART}")
        so = ort.SessionOptions()
        threads = int(os.environ.get("ORT_THREADS", "0"))
        if threads:
            so.intra_op_num_threads = threads
        _sessions[name] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _sessions[name]


def gpt2_artifact() -> str:
    """Preference: weight-only int8 (lossless on our sample, half the size) > fp32 > ORT dynamic int8
    (+0.15 nats on our sample; kept only as a fallback). Override with GPT2_ONNX=<file>."""
    pref = os.environ.get("GPT2_ONNX")
    if pref:
        return pref
    for name in ("gpt2_w8.onnx", "gpt2.onnx", "gpt2_int8.onnx"):
        if os.path.exists(os.path.join(ART, name)):
            return name
    return "gpt2.onnx"


def refiner_artifact() -> str:
    """fp32 by default (0.7 MB, exact); REFINER_ONNX=refiner_encoder_w8.onnx serves the weight-only int8 graph."""
    return os.environ.get("REFINER_ONNX", "refiner_encoder.onnx")


@app.get("/health")
def health():
    present = sorted(f for f in os.listdir(ART) if f.endswith(".onnx")) if os.path.isdir(ART) else []
    return {"status": "ok", "onnxruntime": ort.__version__, "artifacts": present}


class RefinerInput(BaseModel):
    x: list[list[float]] = Field(description="(n, in_dim) node features")
    eigvals: list[float] = Field(description="(d,) Laplacian eigenvalues")
    eigvecs: list[list[float]] = Field(description="(n, d) eigenvectors")
    adj_indices: list[list[int]] = Field(description="(2, E) COO edge index")
    adj_weights: list[float] = Field(description="(E,) edge weights")
    deg_inv: list[float] = Field(description="(n,) inverse degrees")
    return_embeddings: bool = False


@app.post("/refiner/embed")
def refiner_embed(inp: RefinerInput):
    s = session(refiner_artifact())
    feeds = {
        "x": np.asarray(inp.x, dtype=np.float32),
        "eigvals": np.asarray(inp.eigvals, dtype=np.float32),
        "eigvecs": np.asarray(inp.eigvecs, dtype=np.float32),
        "adj_indices": np.asarray(inp.adj_indices, dtype=np.int64),
        "adj_weights": np.asarray(inp.adj_weights, dtype=np.float32),
        "deg_inv": np.asarray(inp.deg_inv, dtype=np.float32),
    }
    if feeds["x"].ndim != 2 or feeds["eigvecs"].shape[0] != feeds["x"].shape[0]:
        raise HTTPException(status_code=422, detail="x must be (n, in_dim) and eigvecs (n, d)")
    try:
        emb = s.run(["embeddings"], feeds)[0]
    except Exception as e:  # shape/dtype errors surface as 422, not 500
        raise HTTPException(status_code=422, detail=str(e)[:300])
    out = {"n": int(emb.shape[0]), "d_model": int(emb.shape[1]), "graph_embedding": emb.mean(axis=0).tolist(), "artifact": refiner_artifact()}
    if inp.return_embeddings:
        out["embeddings"] = emb.tolist()
    return out


class ScoreInput(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


def _tok():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _logits(ids: np.ndarray) -> np.ndarray:
    return session(gpt2_artifact()).run(["logits"], {"input_ids": ids})[0]


@app.post("/gpt2/score")
def gpt2_score(inp: ScoreInput):
    ids = _tok().encode(inp.text)[:1024]
    if len(ids) < 2:
        raise HTTPException(status_code=422, detail="need at least two tokens")
    x = np.array([ids[:-1]], dtype=np.int64); y = np.array(ids[1:])
    logits = _logits(x)[0].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(logits).sum(axis=1))
    nll = lse - logits[np.arange(len(y)), y]
    return {"tokens": len(ids), "mean_nll": float(nll.mean()), "perplexity": float(np.exp(nll.mean())), "artifact": gpt2_artifact()}


class GenerateInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=16, ge=1, le=64)


@app.post("/gpt2/generate")
def gpt2_generate(inp: GenerateInput):
    tok = _tok()
    ids = tok.encode(inp.prompt)[-512:]
    for _ in range(inp.max_new_tokens):  # greedy, full recompute each step (no KV cache in this export)
        nxt = int(_logits(np.array([ids], dtype=np.int64))[0, -1].argmax())
        ids.append(nxt)
    return {"text": tok.decode(ids), "new_tokens": inp.max_new_tokens, "artifact": gpt2_artifact()}
