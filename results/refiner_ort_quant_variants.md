## ORT dynamic quantization variants, refiner encoder, relative output drift vs torch fp32 (n = 1000 probe)

Graph ops: Add×16, Cast×8, Concat×1, Constant×37, ConstantOfShape×2, Div×4, Erf×4, Expand×2, Gather×4, Gemm×17, LayerNormalization×7, MatMul×11, Mul×14, Pow×1, Relu×8, Reshape×2, ScatterElements×2, Shape×6, Slice×2, Squeeze×5, Transpose×2, Unsqueeze×13. 17 Gemm (nn.Linear) and 11 MatMul nodes.

| quantize_dynamic options | drift |
|---|---|
| default (QInt8 weights, dynamic uint8 activations) | 65.36% |
| QUInt8 weights | 65.97% |
| per_channel=True | 65.89% |
| reduce_range=True | 65.83% |
| per_channel + reduce_range | 65.51% |
| Gemm nodes only (the nn.Linear layers) | 0.00% |
| MatMul nodes only | 65.36% |
| exclude PE/in_proj/phi nodes (14) | 65.36% |

Reference: hand-written weight-only int8 on the same model drifts 0.6% (per-tensor) / 0.4% (per-channel) on this probe.
