## GPT-2 (gpt2): mean NLL on the first 4096 tokens of the WikiText-2 (raw) test split (seq 256)

| variant | mean NLL | delta vs torch fp32 | what is quantized |
|---|---|---|---|
| torch_fp32 | 3.8469 | +0.0000 | nothing |
| torch_manual_int8_weight_only | 3.8476 | +0.0006 | 48 block Conv1D weights, per-output-channel int8 (weight-only) |
| ort_fp32 | 3.8470 | +0.0000 | nothing (ONNX Runtime) |
| ort_dynamic_int8 | 4.1838 | +0.3369 | all MatMul weights incl. lm_head to int8 + dynamic int8 activations |

Parity torch vs ORT fp32: max |logit difference| = 7.63e-04. Sizes: fp32 499 MB, int8 126 MB.
