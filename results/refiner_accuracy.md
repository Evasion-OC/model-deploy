## Refiner encoder (spectral_refiner_k16_eps_0_03.pt): relative output drift vs torch fp32 on fixed probe graphs

| n | ORT fp32 | ORT dynamic int8 | manual int8 per-tensor | manual int8 per-channel |
|---|---|---|---|---|
| 300 | 1.8e-07 | 52.78% | 0.53% | 0.32% |
| 1000 | 2.2e-07 | 65.36% | 0.63% | 0.43% |
| 2000 | 2.2e-07 | 61.01% | 0.80% | 0.51% |

ORT dynamic int8 = int8 MatMul weights + dynamically quantized activations; manual = weight-only, 23 tensors. Sizes: fp32 0.72 MB, int8 0.24 MB.
