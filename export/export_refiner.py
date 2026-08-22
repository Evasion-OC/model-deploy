"""Partition-refiner encoder: PyTorch -> ONNX -> ONNX Runtime, with parity, ORT dynamic int8, and the
hand-written weight-only int8 for comparison (same model, two quantization toolchains).
Accuracy metric: relative drift of the encoder output on fixed probe graphs, exactly the proxy the
refiner-perf artifact used, so the numbers are comparable.

    python export/export_refiner.py              # artifacts/refiner_encoder.onnx, artifacts/refiner_encoder_int8.onnx,
                                                 # results/refiner_accuracy.{json,md}
    python export/export_refiner.py --skip-eval  # export only (used by the test fixture)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "export"))
from graph_transformer import SpectralGraphTransformer  # noqa: E402
from quantize_manual import quantized_copy  # noqa: E402

INPUTS = ["x", "eigvals", "eigvecs", "adj_indices", "adj_weights", "deg_inv"]


def build_encoder(ckpt_path):
    """The shipped checkpoint is the full PPO policy; extract its encoder (keys encoder.*)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    enc = SpectralGraphTransformer(in_dim=sd["in_dim"], d_model=sd["d_model"], n_heads=sd["n_heads"],
                                   n_layers=sd["n_layers"], pe_dim=sd["pe_dim"], n_eigs=sd["n_eigs"],
                                   use_local=True, global_kind="lanczos").eval()
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd["model_state"].items() if k.startswith("encoder.")}, strict=True)
    return enc, sd


def probe_graph(n, in_dim, d, seed):
    """Random sparse symmetric graph (~8 edges/node) with real Laplacian eigenpairs (dense eigh, fp64 -> fp32)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    src = torch.randint(0, n, (8 * n,), generator=g); dst = torch.randint(0, n, (8 * n,), generator=g)
    keep = src != dst; src, dst = src[keep], dst[keep]
    ai = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]); aw = torch.ones(ai.shape[1])
    deg = torch.zeros(n).index_add_(0, ai[1], aw).clamp_min(1)
    A = torch.zeros(n, n, dtype=torch.float64); A[ai[0], ai[1]] = 1.0
    ev, V = torch.linalg.eigh(torch.diag(A.sum(1)) - A)
    return x, ev[:d].float(), V[:, :d].float(), ai, aw, 1.0 / deg


def feeds_of(t):
    return {k: v.numpy() for k, v in zip(INPUTS, t)}


def rel_drift(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "refiner", "spectral_refiner_k16_eps_0_03.pt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.results_dir, exist_ok=True)
    enc, sd = build_encoder(a.ckpt)
    in_dim, d = sd["in_dim"], sd["n_eigs"]
    print(f"encoder: {sum(p.numel() for p in enc.parameters()):,} params, in_dim={in_dim}, d_model={sd['d_model']}, n_eigs={d}")

    # 1) export with dynamic n / E (TorchScript-based exporter; index_add_ and matmuls only)
    ex = probe_graph(300, in_dim, d, seed=0)
    path32 = os.path.join(a.out_dir, "refiner_encoder.onnx"); path8 = os.path.join(a.out_dir, "refiner_encoder_int8.onnx")
    with torch.no_grad():
        torch.onnx.export(enc, tuple(ex), path32, input_names=INPUTS, output_names=["embeddings"],
                          dynamic_axes={"x": {0: "n"}, "eigvecs": {0: "n"}, "adj_indices": {1: "E"}, "adj_weights": {0: "E"},
                                        "deg_inv": {0: "n"}, "embeddings": {0: "n"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    quantize_dynamic(path32, path8, weight_type=QuantType.QInt8)
    print(f"exported {path32} ({os.path.getsize(path32)/1e6:.2f} MB), int8 {os.path.getsize(path8)/1e6:.2f} MB")
    if a.skip_eval:
        return

    # 2) parity + quantization comparison on fixed probes
    so = ort.SessionOptions(); so.log_severity_level = 3
    s32 = ort.InferenceSession(path32, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(path8, so, providers=["CPUExecutionProvider"])
    man_pt, nq = quantized_copy(enc, 8, False); man_pc, _ = quantized_copy(enc, 8, True)
    rows = []
    for n, seed in [(300, 1), (1000, 2), (2000, 3)]:
        t = probe_graph(n, in_dim, d, seed)
        with torch.no_grad():
            ref = enc(*t).numpy(); mpt = man_pt(*t).numpy(); mpc = man_pc(*t).numpy()
        o32 = s32.run(["embeddings"], feeds_of(t))[0]; o8 = s8.run(["embeddings"], feeds_of(t))[0]
        rows.append({"n": n, "ort_fp32_vs_torch": rel_drift(o32, ref), "ort_dynamic_int8_vs_torch": rel_drift(o8, ref),
                     "manual_int8_per_tensor_vs_torch": rel_drift(mpt, ref), "manual_int8_per_channel_vs_torch": rel_drift(mpc, ref)})
        print(rows[-1])
    out = {"checkpoint": os.path.basename(a.ckpt), "manual_quantized_tensors": nq, "rows": rows,
           "sizes_mb": {"onnx_fp32": os.path.getsize(path32)/1e6, "onnx_int8": os.path.getsize(path8)/1e6}}
    json.dump(out, open(os.path.join(a.results_dir, "refiner_accuracy.json"), "w"), indent=2)
    with open(os.path.join(a.results_dir, "refiner_accuracy.md"), "w") as f:
        f.write(f"## Refiner encoder ({out['checkpoint']}): relative output drift vs torch fp32 on fixed probe graphs\n\n")
        f.write("| n | ORT fp32 | ORT dynamic int8 | manual int8 per-tensor | manual int8 per-channel |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['n']} | {r['ort_fp32_vs_torch']:.1e} | {r['ort_dynamic_int8_vs_torch']:.2%} | {r['manual_int8_per_tensor_vs_torch']:.2%} | {r['manual_int8_per_channel_vs_torch']:.2%} |\n")
        f.write(f"\nORT dynamic int8 = int8 MatMul weights + dynamically quantized activations; manual = weight-only, {nq} tensors. "
                f"Sizes: fp32 {out['sizes_mb']['onnx_fp32']:.2f} MB, int8 {out['sizes_mb']['onnx_int8']:.2f} MB.\n")
    print("wrote results/refiner_accuracy.{json,md}")


if __name__ == "__main__":
    main()
