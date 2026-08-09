"""
compression/ptq.py
──────────────────
Post-Training Quantisation (PTQ) — 8-bit integer weights.

dynamic_quantize(model)
    Converts nn.Linear weights statically to int8.
    Activations are quantised dynamically at inference time.
    No calibration data needed. Works on CPU.
    → ~4× weight storage reduction for linear layers.

static_quantize(model, calib_loader, num_batches)
    Inserts observer stubs, runs calibration, freezes int8 scale/zero-point
    for both weights and activations. Slightly more accurate than dynamic.
    Requires CPU inference.

Both return a NEW model; the original is untouched.
"""

import copy
import torch
import torch.nn as nn


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """
    Apply dynamic int8 quantisation to all nn.Linear sub-layers.
    Returns a new CPU model.
    """
    q_model = copy.deepcopy(model).cpu().eval()
    q_model = torch.quantization.quantize_dynamic(
        q_model, {nn.Linear}, dtype=torch.qint8,
    )
    print("[ptq] Dynamic int8 quantisation applied  (Linear layers → int8).")
    return q_model


def static_quantize(
    model: nn.Module,
    calib_loader,
    num_batches: int = 10,
) -> nn.Module:
    """
    Apply static int8 quantisation (weights + activations via calibration).
    Returns a new frozen CPU model.
    """
    q_model = copy.deepcopy(model).cpu().eval()

    # Best-effort Conv + BN + ReLU fusion
    try:
        _fuse(q_model)
    except Exception:
        pass

    q_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(q_model, inplace=True)

    print(f"[ptq] Calibrating over {num_batches} batches ...")
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= num_batches:
                break
            q_model(x.cpu())

    torch.quantization.convert(q_model, inplace=True)
    print("[ptq] Static int8 quantisation applied  (weights + activations).")
    return q_model


def _fuse(model: nn.Module) -> None:
    """Fuse Conv → BN → ReLU triplets inside Sequential containers."""
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Sequential):
            continue
        children = list(mod.named_children())
        i = 0
        while i + 2 < len(children):
            n0, m0 = children[i]
            n1, m1 = children[i + 1]
            n2, m2 = children[i + 2]
            if (isinstance(m0, nn.Conv2d) and
                    isinstance(m1, nn.BatchNorm2d) and
                    isinstance(m2, nn.ReLU)):
                prefix = f"{name}." if name else ""
                torch.quantization.fuse_modules(
                    model,
                    [f"{prefix}{n0}", f"{prefix}{n1}", f"{prefix}{n2}"],
                    inplace=True,
                )
                i += 3
            else:
                i += 1
