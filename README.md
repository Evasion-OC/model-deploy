# model-deploy — export, quantize, serve, measure

Two models taken from PyTorch to a served ONNX Runtime container, with every step measured and
two quantization toolchains compared on the same graphs:

- the **partition-refiner encoder** from [multilevel-partition-refinement](https://github.com/Evasion-OC/multilevel-partition-refinement)
  (174k params, graph transformer with a Lanczos spectral mixer; checkpoint shipped here), and
- **GPT-2 124M** (public `gpt2` weights; the same path works for any GPT-2-class checkpoint).

```
PyTorch ──torch.onnx.export──▶ ONNX ──┬─▶ ONNX Runtime fp32           (parity checked vs torch)
                                      ├─▶ ORT quantize_dynamic int8   (the standard tool; measured, then rejected)
                                      └─▶ weight-only int8 via DequantizeLinear (our scheme, as a deployable graph)
                                                        │
                                FastAPI + ONNX Runtime ◀┘   Docker image (no PyTorch inside), tests, CI
```

## Results (CPU, Apple M-series; every number reproducible with the commands below)

### Export parity
| model | exporter | parity vs PyTorch | shapes |
|---|---|---|---|
| refiner encoder | TorchScript-based (`dynamo=False`), opset 17 | relative output drift **2×10⁻⁷** on unseen graphs | dynamic `n` nodes and `E` edges |
| GPT-2 124M | `torch.export`-based (`dynamo=True`), opset 18 | max logit difference **8×10⁻⁴** | batch 1, dynamic sequence length |

**Finding 1 — a silent exporter bug.** The encoder's message passing used `index_add_` with duplicate
destination indices. The TorchScript exporter lowers `index_add_` to a `ScatterElements` *without* a
reduction attribute, so contributions to the same node overwrite instead of summing; the exported
graph ran without any error and drifted **34%** from PyTorch. The fix is the export-friendly equivalent,
`scatter_add`, which lowers with `reduction="add"` (opset ≥ 16); the torch forward is unchanged (verified
equal) and the graph now matches to 2×10⁻⁷. `tests/test_export.py` keeps a duplicate-edge regression test.

### Quantization: two toolchains, same graphs
| model | ORT `quantize_dynamic` int8 | hand-written weight-only int8 (per-channel) |
|---|---|---|
| refiner encoder, output drift vs fp32 | **53–65%** | **0.4–0.6%** |
| GPT-2, mean NLL on a fixed 4,096-token sample (fp32: 4.758) | 4.911 (**+0.153 nats**) | 4.755 (**−0.003**, lossless) |

**Finding 2 — the standard tool quantizes the wrong nodes here.** A variant sweep
(`results/refiner_ort_quant_variants.md`) shows `quantize_dynamic` never touches the 17 `Gemm` nodes that
hold this model's `nn.Linear` weights (restricting it to `Gemm` changes nothing: 0.00% drift) and instead
quantizes the 11 weight-free spectral `MatMul`s (Vᵀh, V·M, (V²)·Φ) with per-tensor uint8 activations —
the eigenvector inputs, whose entries sit around 1/√n, are crushed. No option (`per_channel`,
`reduce_range`, `QUInt8`) changes that. For GPT-2 the same activation quantization costs 0.15 nats.

**Finding 3 — weight-only int8 as a deployable ONNX graph.** `export/weight_only_int8.py` rewrites every
`Gemm`/`MatMul` weight initializer to int8 with a per-output-channel scale behind `DequantizeLinear`
(opset 13), activations untouched — the hand-written scheme from the refiner artifact, now a file ORT
serves. Refiner: 0.72 → 0.22 MB at 0.4–0.6% drift; GPT-2: 499 → 244 MB at −0.003 nats. This buys file size
and memory, not CPU compute: int8 *compute* needs activation quantization, which is exactly what damages
these two models. The server prefers these artifacts.

### Latency
See `results/latency.md` (median of 20, CPU): torch eager vs `torch.compile` vs ORT fp32 vs ORT weight-only
int8 vs ORT dynamic int8, for the encoder at n = 5,000 / 20,000 and GPT-2 at batch 1, seq 128.

## Serving
`serve/app.py` (FastAPI, ONNX Runtime only — the image contains no PyTorch and no model code):

- `GET /health` — runtime version and artifacts present
- `POST /refiner/embed` — node features + eigenpairs + adjacency → node / graph embeddings (fp32 graph by default)
- `POST /gpt2/score` — mean next-token NLL and perplexity of a text (`gpt2_w8.onnx` by default)
- `POST /gpt2/generate` — greedy continuation, ≤ 64 tokens (no KV cache in this export; deliberately simple)

```bash
pip install -r requirements-export.txt
python export/export_refiner.py            # artifacts/refiner_encoder{,_int8}.onnx, results/refiner_accuracy.md
python export/export_gpt2.py               # artifacts/gpt2{,_int8}.onnx, results/gpt2_accuracy.md  (downloads gpt2)
python export/weight_only_int8.py artifacts/refiner_encoder.onnx artifacts/refiner_encoder_w8.onnx
python export/weight_only_int8.py artifacts/gpt2.onnx artifacts/gpt2_w8.onnx
python export/diagnose_ort_quant.py        # results/refiner_ort_quant_variants.md
python bench/latency.py                    # results/latency.md
pytest -q                                  # export parity + API tests
uvicorn serve.app:app --port 8000          # or: docker build -t model-deploy . && docker run -p 8000:8000 model-deploy
```

CI (`.github/workflows/ci.yml`) exports both models, runs the tests, builds the image and curls `/health`.

## Honest scope
- CPU only; GPU execution providers are a one-line change but are not measured here.
- GPT-2 is exported at batch 1 (dynamic sequence) and served without a KV cache; this is a deployment
  study, not a high-throughput LLM server.
- The refiner's accuracy metric is encoder-output drift on fixed probe graphs (the same proxy as the
  quantization artifact); task-level cut quality for the quantized refiner is measured separately in the
  main repository's benchmark.
- Weight-only quantization here covers 2-D `Gemm`/`MatMul` initializers (GPT-2: the 48 block weights; the
  tied embedding/lm-head stays fp32).

Code for the refiner model is copied from the main repository; everything else is new. MIT.
