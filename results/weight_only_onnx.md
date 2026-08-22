## Weight-only int8 as ONNX artifacts (int8 initializers + per-channel DequantizeLinear; activations fp32)

| artifact | size | accuracy vs torch fp32 |
|---|---|---|
| refiner_encoder_w8.onnx | 0.22 MB (fp32 0.72 MB) | output drift 0.44% (n=300), 0.58% (n=1000), 0.60% (n=2000) |
| gpt2_w8.onnx | 244 MB (fp32 499 MB) | mean NLL 4.7548, delta -0.0032 nats |

Compare ORT quantize_dynamic on the same graphs: refiner 65% drift (it quantizes the weight-free spectral MatMuls, never the Gemm weight layers), GPT-2 +0.15 nats. Weight-only keeps the compute in fp32, so it buys file size and memory, not CPU speed; compute-int8 needs activation quantization, which is what damages these models.
