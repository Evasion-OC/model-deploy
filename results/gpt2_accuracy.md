## GPT-2 (gpt2) accuracy on a fixed 4096-token sample (seq 256)

| variant | mean NLL | delta vs torch fp32 | what is quantized |
|---|---|---|---|
| torch_fp32 | 4.7580 | +0.0000 | nothing |
| torch_manual_int8_weight_only | 4.7546 | -0.0034 | 48 block Conv1D weights, per-output-channel int8 (weight-only) |
| ort_fp32 | 4.7580 | +0.0000 | nothing (ONNX Runtime) |
| ort_dynamic_int8 | 4.9110 | +0.1530 | all MatMul weights incl. lm_head to int8 + dynamic int8 activations |

Parity torch vs ORT fp32: max |logit difference| = 7.93e-04. Sizes: fp32 499 MB, int8 126 MB.
