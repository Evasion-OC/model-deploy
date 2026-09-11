## Weight-only int8 ONNX graphs (int8 weights with per-channel DequantizeLinear, fp32 activations)

| artifact | size | accuracy against PyTorch fp32 |
|---|---|---|
| refiner_encoder_w8.onnx | 0.22 MB (fp32 0.72 MB) | output drift 0.44% at n=300, 0.58% at n=1000, 0.60% at n=2000 |
| gpt2_w8.onnx | 244 MB (fp32 499 MB) | mean NLL 3.8484, +0.0015 nats (4096 WikiText-2 test tokens, seq 256) |

Activations stay in fp32. CPU timings for these graphs are in latency.md.
