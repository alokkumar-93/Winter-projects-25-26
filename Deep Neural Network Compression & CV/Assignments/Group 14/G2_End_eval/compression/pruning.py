"""
compression/pruning.py
──────────────────────
Generic global unstructured magnitude pruning.

Works on any model whose layers inherit modified_linear / modified_conv2d.

How it works
────────────
1. Collect absolute values of ALL weights across every compressible layer.
2. Find the k-th smallest value  (k = floor(fraction × N_total)).
3. Zero out the mask for every weight ≤ threshold.
4. Forward pass uses  W_eff = W × mask  — zeroed weights contribute nothing.

Why global (not per-layer)?
────────────────────────────
Per-layer pruning at a fixed ratio over-prunes small but critical layers
(e.g. the final classifier). Global pruning lets each layer's own
distribution determine how many weights are removed.

Public API
──────────
  prune_model(model, fraction)   → float   (threshold used)
  sparsity_stats(model)          → dict    (total, zeros, sparsity_pct)
"""

import torch
import torch.nn as nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d


def _compressible(model: nn.Module):
    """Yield all modified_linear / modified_conv2d modules."""
    for m in model.modules():
        if isinstance(m, (modified_linear, modified_conv2d)):
            yield m


def prune_model(model: nn.Module, prune_fraction: float) -> float:
    """
    Apply global magnitude pruning.

    Parameters
    ----------
    model          : nn.Module containing modified_linear / modified_conv2d
    prune_fraction : fraction of weights to zero out, e.g. 0.50 = 50 %

    Returns
    -------
    threshold : absolute weight value used as the cut-off
    """
    assert 0.0 < prune_fraction < 1.0, "prune_fraction must be in (0, 1)"

    # Gather magnitudes of currently-active (unmasked) weights only.
    # This makes repeated / iterative pruning calls correct.
    parts = []
    for m in _compressible(model):
        w_abs  = m.weight.detach().abs()
        active = w_abs[m.mask.bool()] if m.mask is not None else w_abs.flatten()
        parts.append(active.flatten())

    if not parts:
        raise RuntimeError(
            "No modified_linear / modified_conv2d layers found.\n"
            "Make sure your model uses the project's custom layer wrappers."
        )

    all_w     = torch.cat(parts)
    k         = max(1, int(prune_fraction * all_w.numel()))
    threshold = torch.kthvalue(all_w, k).values.item()

    for m in _compressible(model):
        m.prune(threshold)

    stats = sparsity_stats(model)
    print(
        f"[prune] fraction={prune_fraction:.0%}  "
        f"threshold={threshold:.6f}  "
        f"sparsity={stats['sparsity_pct']:.1f}%  "
        f"({stats['zeros']:,}/{stats['total']:,} weights zeroed)"
    )
    return threshold


def sparsity_stats(model: nn.Module) -> dict:
    """
    Report mask-based sparsity across all compressible layers.

    Returns dict with keys: total, zeros, sparsity_pct
    """
    total = zeros = 0
    for m in _compressible(model):
        total += m.mask.numel()
        zeros += (m.mask == 0).sum().item()
    pct = 100.0 * zeros / total if total else 0.0
    return {"total": total, "zeros": zeros, "sparsity_pct": round(pct, 2)}
