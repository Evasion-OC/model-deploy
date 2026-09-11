# model-deploy

Exporting two PyTorch models to ONNX Runtime, quantizing them, serving them, and measuring each step.
The models are the graph transformer encoder from
[multilevel-partition-refinement](https://github.com/Evasion-OC/multilevel-partition-refinement)
(174k parameters, with a Lanczos spectral mixer; the checkpoint is included here) and GPT-2 124M
(the public `gpt2` weights).

For each model the repo exports an ONNX graph and checks it against PyTorch, compares ONNX Runtime's
dynamic int8 quantization with a hand-written weight-only int8 scheme, times the variants on CPU, and
serves them from a FastAPI app in a Docker image that contains ONNX Runtime and the exported graphs
but no PyTorch.

## Results

Timings are from an Apple M1 Pro CPU. Every table is produced by the commands under "Reproducing".

### Export parity

| model | exporter | difference from PyTorch | dynamic shapes |
|---|---|---|---|
| refiner encoder | TorchScript-based (`dynamo=False`), opset 17 | relative output difference 2e-7 on graphs not used for the export | nodes n, edges E |
| GPT-2 124M | `torch.export`-based (`dynamo=True`), opset 18 | max logit difference 8e-4 | sequence length (batch 1) |

The first export of the encoder ran without errors but its output differed from PyTorch by 34%.
The message-passing layer used `index_add_`, which the TorchScript exporter turns into a
`ScatterElements` node with no reduction, so edges that share a destination node overwrite each other
instead of adding up. `scatter_add` gives the same result in PyTorch and exports with
`reduction="add"` (opset 16 and later), which brings the difference down to 2e-7.
`tests/test_export.py` keeps a regression test with duplicated edges.

### Quantization

| | ONNX Runtime `quantize_dynamic` (int8) | hand-written weight-only int8, per channel |
|---|---|---|
| refiner encoder: output drift from fp32 | 53 to 65% | 0.4 to 0.6% |
| GPT-2: mean NLL on 4,096 WikiText-2 test tokens (fp32 3.847) | 4.184 (+0.34 nats) | 3.848 (+0.002) |

`quantize_dynamic` leaves the 17 `Gemm` nodes that hold the encoder's `nn.Linear` weights untouched:
restricting it to `Gemm` nodes gives 0.00% drift. The error comes from the 11 `MatMul` nodes that carry
no weights (the products with the eigenvectors), which it quantizes with per-tensor uint8 activations;
restricting it to those nodes reproduces the full drift. Its options (`per_channel`, `reduce_range`,
`QUInt8`) make no difference (`results/refiner_ort_quant_variants.md`). On GPT-2 the same
activation quantization costs 0.34 nats.

`export/weight_only_int8.py` writes the hand-written scheme into the graph itself: each `Gemm` or
`MatMul` weight becomes an int8 tensor with one scale per output channel, followed by a
`DequantizeLinear` node, and activations stay in fp32. The encoder goes from 0.72 MB to 0.22 MB at 0.4
to 0.6% drift, and GPT-2 from 499 MB to 244 MB at +0.0015 nats (`results/weight_only_onnx.md`).
The server uses these graphs by default.

### Latency

| | PyTorch eager | ORT fp32 | ORT weight-only int8 | ORT dynamic int8 |
|---|---|---|---|---|
| refiner encoder, n = 5,000 | 22.6 ms | 55.5 ms | 56.1 ms | 53.8 ms |
| refiner encoder, n = 20,000 | 86.2 ms | 233.3 ms | 235.0 ms | 224.3 ms |
| GPT-2 124M, batch 1, 128 tokens | 65.5 ms | 78.5 ms | 51.7 ms | 30.9 ms |

Median of 20 runs after warm-up, 8 threads (`results/latency.md`). The two models behave differently.
For the encoder, ONNX Runtime is 2.5 to 2.7 times slower than PyTorch eager on this CPU, and int8
changes nothing because its weight layers are never quantized. For GPT-2, dynamic int8 is 2.1 times
faster than PyTorch eager, at the accuracy cost above, and the weight-only graph is 1.3 times faster
with no accuracy cost. `torch.compile` is missing from the table because its C++ backend stalled on
this machine.

## Serving

`serve/app.py` runs everything in ONNX Runtime:

- `GET /health`: runtime version and the artifacts present
- `POST /refiner/embed`: node features, eigenpairs and adjacency in; node and graph embeddings out
- `POST /gpt2/score`: mean next-token NLL and perplexity of a text
- `POST /gpt2/generate`: greedy continuation, up to 64 tokens

## Reproducing

```bash
pip install -r requirements-export.txt
python export/export_refiner.py         # refiner ONNX graphs, results/refiner_accuracy.md
python export/export_gpt2.py            # GPT-2 ONNX graphs, results/gpt2_accuracy.md
python export/weight_only_int8.py artifacts/refiner_encoder.onnx artifacts/refiner_encoder_w8.onnx
python export/weight_only_int8.py artifacts/gpt2.onnx artifacts/gpt2_w8.onnx
python export/eval_weight_only.py       # results/weight_only_onnx.md
python export/diagnose_ort_quant.py     # results/refiner_ort_quant_variants.md
python bench/latency.py --skip-compile  # results/latency.md
pytest -q
uvicorn serve.app:app --port 8000       # or: docker build -t model-deploy . && docker run -p 8000:8000 model-deploy
```

CI (`.github/workflows/ci.yml`) exports both models, runs the tests, builds the image and checks
`/health`.

## Limitations

- CPU only, and the timings come from one laptop.
- GPT-2 is exported for batch 1 with variable sequence length and served without a KV cache, so
  generation recomputes the whole sequence at each step.
- The encoder's accuracy is measured as output drift on fixed probe graphs, not as partitioning quality.
- The weight-only scheme covers the 2-D `Gemm` and `MatMul` weights. GPT-2's tied embedding and output
  layer stay in fp32.

The refiner model code is copied from the main repository.
