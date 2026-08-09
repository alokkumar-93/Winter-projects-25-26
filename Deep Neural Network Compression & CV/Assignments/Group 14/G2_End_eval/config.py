"""
config.py
─────────
Single source of truth for every hyperparameter.
Edit here — nothing else in the codebase needs to change.
"""

import os
import torch


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[config] device: {device}")
    return device


# ── Data ──────────────────────────────────────────────────────────────────────

DATA_DIR        = "data"
BATCH_SIZE      = 128
NUM_WORKERS     = 2
AUGMENT         = True          # random crop + horizontal flip


# ── Training ──────────────────────────────────────────────────────────────────

EPOCHS_BASELINE = 20            # fresh training
EPOCHS_FINETUNE = 10            # after each pruning step
LR_BASELINE     = 0.01
LR_FINETUNE     = 0.005         # lower lr for sparse fine-tuning
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
LR_SCHEDULER    = "cosine"      # "cosine" | "step" | None


# ── Pruning ───────────────────────────────────────────────────────────────────

PRUNE_RATIOS           = [0.30, 0.50, 0.70]   # benchmarked together
STRUCTURED_PRUNE       = True
STRUCTURED_PRUNE_RATIO = 0.25    # fraction of conv filters removed per layer


# ── Quantisation ─────────────────────────────────────────────────────────────

KMEANS_K        = 16            # codebook size for KMeans weight quantisation
PTQ_DTYPE       = torch.qint8   # dtype for post-training dynamic quantisation


# ── Paths ─────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"

for _d in (CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)
