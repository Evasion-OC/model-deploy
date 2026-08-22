"""The hand-written weight-only symmetric fake quantizer from the refiner artifact (refiner-perf),
copied verbatim so the comparison against ONNX Runtime's quantizer is against exactly that scheme."""
import torch


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def fake_quantize(W: torch.Tensor, bits: int = 8, per_channel: bool = False) -> torch.Tensor:
    """Symmetric fake quantization. per_channel scales along dim 0 (out-channels)."""
    Q = qmax(bits)
    if per_channel and W.dim() >= 2:
        amax = W.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
        s = (amax / Q).view(-1, *([1] * (W.dim() - 1)))
    else:
        s = (W.abs().max().clamp_min(1e-12) / Q)
    return (W / s).round().clamp(-Q, Q) * s


def quantized_copy(module: torch.nn.Module, bits: int, per_channel: bool):
    import copy
    m = copy.deepcopy(module); n = 0
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("weight") and p.dim() >= 2:
                p.copy_(fake_quantize(p, bits, per_channel)); n += 1
    return m, n
