"""CPU latency: torch eager vs torch.compile vs ONNX Runtime fp32 vs ORT dynamic int8, for both models.
Median of N timed runs after warm-up, single process, threads reported. Writes results/latency.md.

    python bench/latency.py --runs 20
"""
import argparse, os, platform, statistics, sys, time
import numpy as np
import torch
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "export"))
from export_refiner import build_encoder, INPUTS  # noqa: E402


def timed(fn, runs, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def sess(path):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def latency_inputs(n, in_dim, d, seed=0):
    """Random orthonormal V via QR (latency does not need true eigenpairs); ~8 edges/node."""
    g = torch.Generator().manual_seed(seed)
    V, _ = torch.linalg.qr(torch.randn(n, d, generator=g))
    src = torch.randint(0, n, (8 * n,), generator=g); dst = torch.randint(0, n, (8 * n,), generator=g)
    ai = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]); aw = torch.ones(ai.shape[1])
    deg = torch.zeros(n).index_add_(0, ai[1], aw).clamp_min(1)
    return torch.randn(n, in_dim, generator=g), torch.sort(torch.rand(d, generator=g))[0], V, ai, aw, 1.0 / deg


def compile_or_none(model):
    try:
        m = torch.compile(model)
        return m
    except Exception as e:  # pragma: no cover
        print(f"torch.compile unavailable: {e}"); return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=20); ap.add_argument("--seq", type=int, default=128)
    a = ap.parse_args()
    art = os.path.join(ROOT, "artifacts"); lines = []
    hdr = f"platform {platform.machine()} / {platform.system()}, torch {torch.__version__} ({torch.get_num_threads()} threads), onnxruntime {ort.__version__}"
    print(hdr); lines.append(f"## CPU latency (median of {a.runs}, ms)\n\n{hdr}\n")

    # --- refiner encoder ---
    enc, sd = build_encoder(os.path.join(ROOT, "models", "refiner", "spectral_refiner_k16_eps_0_03.pt"))
    s32, s8 = sess(os.path.join(art, "refiner_encoder.onnx")), sess(os.path.join(art, "refiner_encoder_int8.onnx"))
    sw8 = sess(os.path.join(art, "refiner_encoder_w8.onnx"))
    enc_c = compile_or_none(enc)
    lines.append("| refiner encoder | torch eager | torch.compile | ORT fp32 | ORT weight-only int8 | ORT dynamic int8 |\n|---|---|---|---|---|---|")
    for n in (5000, 20000):
        t = latency_inputs(n, sd["in_dim"], sd["n_eigs"]); feeds = {k: v.numpy() for k, v in zip(INPUTS, t)}
        with torch.no_grad():
            e = timed(lambda: enc(*t), a.runs)
            c = timed(lambda: enc_c(*t), a.runs) if enc_c is not None else float("nan")
        o32 = timed(lambda: s32.run(None, feeds), a.runs); o8 = timed(lambda: s8.run(None, feeds), a.runs)
        ow8 = timed(lambda: sw8.run(None, feeds), a.runs)
        row = f"| n = {n:,} | {e:.1f} | {c:.1f} | {o32:.1f} | {ow8:.1f} | {o8:.1f} |"; print(row); lines.append(row)

    # --- GPT-2 ---
    g32, g8 = os.path.join(art, "gpt2.onnx"), os.path.join(art, "gpt2_int8.onnx")
    if os.path.exists(g32):
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
        ids = torch.randint(0, 50257, (1, a.seq)); feeds = {"input_ids": ids.numpy()}
        s32, s8 = sess(g32), sess(g8); sw8 = sess(os.path.join(art, "gpt2_w8.onnx"))
        m_c = compile_or_none(m)
        with torch.no_grad():
            e = timed(lambda: m(input_ids=ids, use_cache=False).logits, a.runs)
            c = timed(lambda: m_c(input_ids=ids, use_cache=False).logits, a.runs) if m_c is not None else float("nan")
        o32 = timed(lambda: s32.run(None, feeds), a.runs); o8 = timed(lambda: s8.run(None, feeds), a.runs)
        ow8 = timed(lambda: sw8.run(None, feeds), a.runs)
        lines.append(f"\n| GPT-2 124M, batch 1, seq {a.seq} | torch eager | torch.compile | ORT fp32 | ORT weight-only int8 | ORT dynamic int8 |\n|---|---|---|---|---|---|")
        row = f"| one forward | {e:.1f} | {c:.1f} | {o32:.1f} | {ow8:.1f} | {o8:.1f} |"; print(row); lines.append(row)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    open(os.path.join(ROOT, "results", "latency.md"), "w").write("\n".join(lines) + "\n")
    print("wrote results/latency.md")


if __name__ == "__main__":
    main()
