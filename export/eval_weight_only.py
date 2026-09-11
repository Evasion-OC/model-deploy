"""Accuracy of the weight-only int8 ONNX graphs (built by weight_only_int8.py) against PyTorch fp32.

Refiner: relative output drift on the same fixed probe graphs as export_refiner.py.
GPT-2: mean next-token NLL on the same tokens as export_gpt2.py (its results JSON supplies the fp32
baseline and the evaluation settings, so run export_gpt2.py first).

    python export/eval_weight_only.py        # writes results/weight_only_onnx.{json,md}
"""
import json, os, sys
import numpy as np
import torch
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "export"))
from export_refiner import build_encoder, probe_graph, feeds_of, rel_drift  # noqa: E402

ART, RES = os.path.join(ROOT, "artifacts"), os.path.join(ROOT, "results")


def session(path):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    out = {}
    enc, sd = build_encoder(os.path.join(ROOT, "models", "refiner", "spectral_refiner_k16_eps_0_03.pt"))
    s = session(os.path.join(ART, "refiner_encoder_w8.onnx"))
    rows = []
    for n, seed in [(300, 1), (1000, 2), (2000, 3)]:
        t = probe_graph(n, sd["in_dim"], sd["n_eigs"], seed)
        with torch.no_grad():
            ref = enc(*t).numpy()
        rows.append({"n": n, "drift": rel_drift(s.run(["embeddings"], feeds_of(t))[0], ref)})
    out["refiner"] = {"rows": rows, "size_mb": os.path.getsize(os.path.join(ART, "refiner_encoder_w8.onnx")) / 1e6,
                      "fp32_size_mb": os.path.getsize(os.path.join(ART, "refiner_encoder.onnx")) / 1e6}

    g8 = os.path.join(ART, "gpt2_w8.onnx"); acc_path = os.path.join(RES, "gpt2_accuracy.json")
    if os.path.exists(g8) and os.path.exists(acc_path):
        from transformers import GPT2TokenizerFast
        from export_gpt2 import eval_text, nll_ort
        acc = json.load(open(acc_path))
        tok = GPT2TokenizerFast.from_pretrained(acc["model"])
        ids = tok.encode(eval_text())[: acc["eval_tokens"]]
        nll = nll_ort(session(g8), ids, acc["seq"]); base = acc["nll"]["torch_fp32"]
        out["gpt2"] = {"nll": nll, "delta_vs_torch_fp32": nll - base, "eval_tokens": len(ids), "seq": acc["seq"],
                       "size_mb": os.path.getsize(g8) / 1e6, "fp32_size_mb": os.path.getsize(os.path.join(ART, "gpt2.onnx")) / 1e6}

    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "weight_only_onnx.json"), "w"), indent=2)
    r = out["refiner"]
    lines = ["## Weight-only int8 ONNX graphs (int8 weights with per-channel DequantizeLinear, fp32 activations)", "",
             "| artifact | size | accuracy against PyTorch fp32 |", "|---|---|---|",
             f"| refiner_encoder_w8.onnx | {r['size_mb']:.2f} MB (fp32 {r['fp32_size_mb']:.2f} MB) | output drift "
             + ", ".join(f"{x['drift']:.2%} at n={x['n']}" for x in r["rows"]) + " |"]
    if "gpt2" in out:
        g = out["gpt2"]
        lines.append(f"| gpt2_w8.onnx | {g['size_mb']:.0f} MB (fp32 {g['fp32_size_mb']:.0f} MB) | mean NLL {g['nll']:.4f}, "
                     f"{g['delta_vs_torch_fp32']:+.4f} nats ({g['eval_tokens']} WikiText-2 test tokens, seq {g['seq']}) |")
    lines += ["", "Activations stay in fp32. CPU timings for these graphs are in latency.md."]
    open(os.path.join(RES, "weight_only_onnx.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
