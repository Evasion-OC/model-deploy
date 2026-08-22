"""Weight-only int8 as an ONNX artifact: every Gemm/MatMul weight initializer becomes an int8 tensor with a
per-output-channel scale, rewired through DequantizeLinear (opset 13+). Activations stay fp32, so this is
exactly the hand-written scheme from the refiner artifact, now in a file ONNX Runtime can serve.

ORT's quantize_dynamic cannot produce this: it quantizes activations too (MatMulInteger) and, for graphs
exported from nn.Linear, skips the Gemm nodes that hold the weights.

    python export/weight_only_int8.py artifacts/refiner_encoder.onnx artifacts/refiner_encoder_w8.onnx
    python export/weight_only_int8.py artifacts/gpt2.onnx artifacts/gpt2_w8.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def quantize_weights(src, dst, min_numel=1024):
    m = onnx.load(src, load_external_data=True)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    consumers = {}
    for node in g.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    done, new_nodes = 0, []
    for name, t in list(inits.items()):
        users = consumers.get(name, [])
        if not users or any(n.op_type not in ("Gemm", "MatMul") for n in users):
            continue
        W = numpy_helper.to_array(t)
        if W.dtype != np.float32 or W.ndim != 2 or W.size < min_numel:
            continue
        # output-channel axis: Gemm with transB=1 stores (out, in) -> axis 0; MatMul (in, out) -> axis 1
        n0 = users[0]
        transB = any(a.name == "transB" and a.i == 1 for a in n0.attribute) if n0.op_type == "Gemm" else False
        axis = 0 if (n0.op_type == "Gemm" and transB) else 1
        amax = np.maximum(np.abs(W).max(axis=1 - axis, keepdims=True), 1e-12)
        scale = (amax / 127.0).astype(np.float32)
        Wq = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
        g.initializer.remove(t)
        g.initializer.extend([
            numpy_helper.from_array(Wq, name + "_q"),
            numpy_helper.from_array(scale.reshape(-1), name + "_scale"),
            numpy_helper.from_array(np.zeros(scale.size, dtype=np.int8), name + "_zp"),
        ])
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"], [name],
                                          name=f"dq_{name}", axis=axis))
        done += 1
    # DequantizeLinear nodes must precede their consumers: prepend them
    all_nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(all_nodes)
    # opset >= 13 for per-axis DequantizeLinear
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < 13:
            op.version = 13
    onnx.save(m, dst)
    return done


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = quantize_weights(src, dst)
    import os
    print(f"{dst}: {n} weight tensors -> int8 per-output-channel; {os.path.getsize(src)/1e6:.1f} MB -> {os.path.getsize(dst)/1e6:.1f} MB")
