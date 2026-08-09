"""
compression/structured_pruning.py
──────────────────────────────────
Structured (filter-level) pruning for Conv2d layers.

Why structured pruning?
───────────────────────
Unstructured pruning scatters zeros randomly through weight tensors.
Standard hardware doesn't skip zeros in dense matrix multiplications,
so there's no latency benefit without specialised sparse kernels.

Structured pruning removes ENTIRE output filters (output channels)
whose L1 norm is weakest. This creates zero-filled rows in the weight
tensor — hardware-friendly and more compressible in storage.

Implementation
──────────────
We zero the full filter slice in the layer's mask buffer (same interface
as unstructured pruning) rather than physically reshaping the architecture.
Physical channel removal would require rewriting subsequent BN/Conv shapes.

Public API
──────────
  prune_filters(model, fraction)   → int    (total filters zeroed)
  filter_sparsity(model)           → dict
"""

import torch
import torch.nn as nn
from compression.conv2d import modified_conv2d


def prune_filters(model: nn.Module, fraction: float = 0.25) -> int:
    """
    Zero out the weakest `fraction` of output filters per Conv2d layer.

    Weakness = L1 norm of the filter (sum of |w| over in_ch × kH × kW).

    Parameters
    ----------
    model    : nn.Module with modified_conv2d layers
    fraction : fraction of filters to remove per layer

    Returns
    -------
    total_zeroed : total number of filters zeroed across all conv layers
    """
    assert 0.0 < fraction < 1.0

    total_zeroed = 0

    for name, m in model.named_modules():
        if not isinstance(m, modified_conv2d):
            continue

        # weight: (out_channels, in_channels, kH, kW)
        w            = m.weight.detach()
        n_filters    = w.shape[0]
        filter_norms = w.abs().sum(dim=(1, 2, 3))   # (out_channels,)

        k         = max(1, int(fraction * n_filters))
        threshold = torch.kthvalue(filter_norms, k).values.item()

        weak = (filter_norms <= threshold)           # bool (out_channels,)
        m.mask.data[weak] = 0.0                      # zero entire filter slices

        n_zeroed      = int(weak.sum().item())
        total_zeroed += n_zeroed
        print(
            f"  [struct-prune] {name:35s} "
            f"{n_filters} filters  → zeroed {n_zeroed} "
            f"({100 * n_zeroed / n_filters:.1f}%)"
        )

    print(f"[struct-prune] total filters zeroed: {total_zeroed}")
    return total_zeroed


def filter_sparsity(model: nn.Module) -> dict:
    """
    Report structured sparsity (how many full filters are completely zeroed).

    Returns dict with keys: total_filters, zeroed_filters, filter_sparsity_pct
    """
    total_filters = zeroed_filters = 0

    for m in model.modules():
        if isinstance(m, modified_conv2d):
            n_out = m.mask.shape[0]
            total_filters += n_out
            for f_idx in range(n_out):
                if m.mask[f_idx].sum().item() == 0.0:
                    zeroed_filters += 1

    pct = 100.0 * zeroed_filters / total_filters if total_filters else 0.0
    return {
        "total_filters":       total_filters,
        "zeroed_filters":      zeroed_filters,
        "filter_sparsity_pct": round(pct, 2),
    }
