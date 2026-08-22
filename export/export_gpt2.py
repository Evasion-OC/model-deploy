"""GPT-2 (124M): PyTorch -> ONNX -> ONNX Runtime, with parity, ORT dynamic int8, and a hand-written
weight-only int8 for comparison. Accuracy metric: mean next-token NLL on a fixed text sample
(export/eval_text.txt, our own READMEs), so every variant is scored on identical tokens.

    python export/export_gpt2.py                 # writes artifacts/gpt2.onnx, artifacts/gpt2_int8.onnx,
                                                 # results/gpt2_accuracy.{json,md}
    python export/export_gpt2.py --model gpt2 --seq 256
"""
import argparse, json, os, time
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, input_ids):
        return self.m(input_ids=input_ids, use_cache=False).logits


def fake_quantize_per_out_channel(W, bits=8, out_dim=1):
    """Symmetric weight-only fake quant with one scale per OUTPUT channel (same scheme as the
    refiner artifact's quantizer). GPT-2's Conv1D stores weights as (in, out), so out_dim=1."""
    Q = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=0 if out_dim == 1 else 1, keepdim=True).clamp_min(1e-12)
    s = amax / Q
    return (torch.clamp(torch.round(W / s), -Q, Q) * s).to(W.dtype)


def manual_int8_copy(model):
    """Quantize the transformer blocks' Conv1D weights (attn + mlp); embeddings and the tied lm_head stay fp32."""
    import copy
    m = copy.deepcopy(model)
    n = 0
    for name, mod in m.named_modules():
        if mod.__class__.__name__ == "Conv1D" and ".h." in name:
            with torch.no_grad():
                mod.weight.copy_(fake_quantize_per_out_channel(mod.weight, 8, out_dim=1)); n += 1
    return m, n


def chunks(ids, seq):
    for i in range(0, len(ids) - 1, seq):
        c = ids[i:i + seq + 1]
        if len(c) > 1:
            yield c


def nll_torch(model, ids, seq):
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for c in chunks(ids, seq):
            x = torch.tensor([c[:-1]]); y = torch.tensor([c[1:]])
            logits = model(input_ids=x, use_cache=False).logits[0].float()
            tot += torch.nn.functional.cross_entropy(logits, y[0], reduction="sum").item(); cnt += y.numel()
    return tot / cnt


def nll_ort(sess, ids, seq):
    tot, cnt = 0.0, 0
    for c in chunks(ids, seq):
        x = np.array([c[:-1]], dtype=np.int64); y = np.array(c[1:])
        logits = sess.run(["logits"], {"input_ids": x})[0][0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        lse = np.log(np.exp(logits).sum(axis=1))
        tot += float((lse - logits[np.arange(len(y)), y]).sum()); cnt += len(y)
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--text", default="export/eval_text.txt")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.results_dir, exist_ok=True)
    torch.manual_seed(0)

    tok = GPT2TokenizerFast.from_pretrained(a.model)
    model = GPT2LMHeadModel.from_pretrained(a.model, attn_implementation="eager").eval()
    ids = tok.encode(open(a.text, encoding="utf-8").read())[: a.max_tokens]
    print(f"{a.model}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params; eval sample {len(ids)} tokens, seq {a.seq}")

    # 1) export (TorchScript-based exporter, dynamic batch/seq)
    path32 = os.path.join(a.out_dir, "gpt2.onnx"); path8 = os.path.join(a.out_dir, "gpt2_int8.onnx")
    t0 = time.time()
    with torch.no_grad():
        # transformers >= 5 builds attention masks through utilities the TorchScript tracer cannot follow;
        # the dynamo (torch.export) path handles them. Dynamic batch/seq via dynamic_shapes.
        ex = torch.tensor([ids[:16]])
        prog = torch.onnx.export(LogitsOnly(model), (ex,), input_names=["input_ids"], output_names=["logits"],
                                 dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=1, max=64),
                                                               1: torch.export.Dim("seq", min=2, max=1024)}},
                                 opset_version=18, dynamo=True)
        prog.optimize(); prog.save(path32)
    print(f"exported {path32} ({os.path.getsize(path32)/1e6:.0f} MB) in {time.time()-t0:.0f}s")

    # 2) ORT dynamic int8 (int8 weights for MatMul + dynamically quantized activations)
    t0 = time.time()
    quantize_dynamic(path32, path8, weight_type=QuantType.QInt8)
    print(f"ORT dynamic int8 -> {path8} ({os.path.getsize(path8)/1e6:.0f} MB) in {time.time()-t0:.0f}s")

    # 3) parity + accuracy
    so = ort.SessionOptions(); so.log_severity_level = 3
    s32 = ort.InferenceSession(path32, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(path8, so, providers=["CPUExecutionProvider"])
    x = torch.tensor([ids[:64]])
    with torch.no_grad():
        lt = model(input_ids=x, use_cache=False).logits[0].numpy()
    lo = s32.run(["logits"], {"input_ids": x.numpy()})[0][0]
    parity = float(np.abs(lt - lo).max())
    print(f"parity torch vs ORT fp32: max|dlogit| = {parity:.2e}")

    man, nq = manual_int8_copy(model)
    rows = {
        "torch_fp32": nll_torch(model, ids, a.seq),
        "torch_manual_int8_weight_only": nll_torch(man, ids, a.seq),
        "ort_fp32": nll_ort(s32, ids, a.seq),
        "ort_dynamic_int8": nll_ort(s8, ids, a.seq),
    }
    base = rows["torch_fp32"]
    out = {"model": a.model, "eval_tokens": len(ids), "seq": a.seq, "parity_max_abs_logit_diff": parity,
           "manual_quantized_tensors": nq, "nll": rows, "nll_delta_vs_torch_fp32": {k: v - base for k, v in rows.items()},
           "sizes_mb": {"onnx_fp32": os.path.getsize(path32)/1e6, "onnx_int8": os.path.getsize(path8)/1e6}}
    json.dump(out, open(os.path.join(a.results_dir, "gpt2_accuracy.json"), "w"), indent=2)
    with open(os.path.join(a.results_dir, "gpt2_accuracy.md"), "w") as f:
        f.write(f"## GPT-2 ({a.model}) accuracy on a fixed {len(ids)}-token sample (seq {a.seq})\n\n")
        f.write("| variant | mean NLL | delta vs torch fp32 | what is quantized |\n|---|---|---|---|\n")
        what = {"torch_fp32": "nothing", "torch_manual_int8_weight_only": f"{nq} block Conv1D weights, per-output-channel int8 (weight-only)",
                "ort_fp32": "nothing (ONNX Runtime)", "ort_dynamic_int8": "all MatMul weights incl. lm_head to int8 + dynamic int8 activations"}
        for k, v in rows.items():
            f.write(f"| {k} | {v:.4f} | {v-base:+.4f} | {what[k]} |\n")
        f.write(f"\nParity torch vs ORT fp32: max |logit difference| = {parity:.2e}. Sizes: fp32 {out['sizes_mb']['onnx_fp32']:.0f} MB, int8 {out['sizes_mb']['onnx_int8']:.0f} MB.\n")
    print(json.dumps(out["nll"], indent=2)); print("wrote results/gpt2_accuracy.{json,md}")


if __name__ == "__main__":
    main()
