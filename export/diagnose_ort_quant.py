"""Why does ONNX Runtime's dynamic int8 drift the refiner encoder by >50% when weight-only int8 drifts <1%?
Sweep the quantizer's options on the same graph and measure output drift vs torch fp32 on a fixed probe.
Writes results/refiner_ort_quant_variants.md.
"""
import os, sys, tempfile
import numpy as np, onnx, torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "export"))
from export_refiner import build_encoder, probe_graph, feeds_of, rel_drift, INPUTS  # noqa: E402

path32 = os.path.join(ROOT, "artifacts", "refiner_encoder.onnx")
enc, sd = build_encoder(os.path.join(ROOT, "models", "refiner", "spectral_refiner_k16_eps_0_03.pt"))
t = probe_graph(1000, sd["in_dim"], sd["n_eigs"], 2)
with torch.no_grad():
    ref = enc(*t).numpy()
m = onnx.load(path32)
ops = {}
for node in m.graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1
gemm_names = [n.name for n in m.graph.node if n.op_type == "Gemm"]
matmul_names = [n.name for n in m.graph.node if n.op_type == "MatMul"]
print("ops in graph:", {k: v for k, v in sorted(ops.items())})
print(f"{len(gemm_names)} Gemm (nn.Linear) nodes, {len(matmul_names)} MatMul nodes")


def drift_for(**kw):
    with tempfile.TemporaryDirectory() as d:
        p8 = os.path.join(d, "q.onnx"); quantize_dynamic(path32, p8, **kw)
        so = ort.SessionOptions(); so.log_severity_level = 3
        s = ort.InferenceSession(p8, so, providers=["CPUExecutionProvider"])
        return rel_drift(s.run(["embeddings"], feeds_of(t))[0], ref)


variants = [
    ("default (QInt8 weights, dynamic uint8 activations)", dict(weight_type=QuantType.QInt8)),
    ("QUInt8 weights", dict(weight_type=QuantType.QUInt8)),
    ("per_channel=True", dict(weight_type=QuantType.QInt8, per_channel=True)),
    ("reduce_range=True", dict(weight_type=QuantType.QInt8, reduce_range=True)),
    ("per_channel + reduce_range", dict(weight_type=QuantType.QInt8, per_channel=True, reduce_range=True)),
    ("Gemm nodes only (the nn.Linear layers)", dict(weight_type=QuantType.QInt8, op_types_to_quantize=["Gemm"])),
    ("MatMul nodes only", dict(weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])),
]
# exclude the layers that touch raw spectral inputs: the PE MLPs and the first projection
spectral_nodes = [n.name for n in m.graph.node if n.op_type in ("Gemm", "MatMul") and any(k in n.name for k in ("pe", "in_proj", "phi"))]
variants.append((f"exclude PE/in_proj/phi nodes ({len(spectral_nodes)})", dict(weight_type=QuantType.QInt8, nodes_to_exclude=spectral_nodes)))
rows = []
for name, kw in variants:
    try:
        d = drift_for(**kw); rows.append((name, f"{d:.2%}")); print(f"{name:55s} drift {d:.2%}")
    except Exception as e:
        rows.append((name, f"error: {str(e)[:60]}")); print(f"{name:55s} error {e}")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "refiner_ort_quant_variants.md"), "w") as f:
    f.write("## ORT dynamic quantization variants, refiner encoder, relative output drift vs torch fp32 (n = 1000 probe)\n\n")
    f.write(f"Graph ops: {', '.join(f'{k}×{v}' for k, v in sorted(ops.items()))}. {len(gemm_names)} Gemm (nn.Linear) and {len(matmul_names)} MatMul nodes.\n\n")
    f.write("| quantize_dynamic options | drift |\n|---|---|\n")
    for n_, d_ in rows:
        f.write(f"| {n_} | {d_} |\n")
    f.write("\nReference: hand-written weight-only int8 on the same model drifts 0.6% (per-tensor) / 0.4% (per-channel) on this probe.\n")
print("wrote results/refiner_ort_quant_variants.md")
