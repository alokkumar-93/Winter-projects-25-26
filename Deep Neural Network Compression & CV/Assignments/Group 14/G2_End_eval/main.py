"""
main.py
────────
Original project entry point — MNIST demo using the full compression flow:
    train → prune → fine-tune → quantise (KMeans) → save .npz → reload

For the full CIFAR-10 multi-ratio benchmark run:
    python benchmark.py
"""

import os
import torch

from config              import get_device
from models.mnist        import mnist_model
from data.data_loader    import MNIST_loader
from utils.training      import train_and_eval, evaluate
from compression.pruning import prune_model, sparsity_stats
from utils.loading       import save_model_npz, load_model_from_npz, load_csr_from_npz

os.makedirs("checkpoints", exist_ok=True)

device = get_device()

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader, test_loader = MNIST_loader(path="data")
model = mnist_model().to(device)

# ── Train baseline ────────────────────────────────────────────────────────────
print("\n── [1] Baseline training ──")
train_and_eval(model, train_loader, test_loader, device, epochs=5, label="baseline")

# ── Prune ─────────────────────────────────────────────────────────────────────
print("\n── [2] Pruning (95%) ──")
prune_model(model, prune_fraction=0.95)
print(sparsity_stats(model))

# ── Fine-tune ─────────────────────────────────────────────────────────────────
print("\n── [3] Fine-tuning ──")
train_and_eval(model, train_loader, test_loader, device,
               epochs=3, lr=0.005, label="post-prune")

# ── Save dense checkpoint ─────────────────────────────────────────────────────
torch.save(model.state_dict(), "checkpoints/mnist_pruned.pth")
print("  Saved → checkpoints/mnist_pruned.pth")

# ── KMeans quantisation (16 centroids) ───────────────────────────────────────
print("\n── [4] KMeans quantisation (K=16) ──")
model.quantize(16)
train_and_eval(model, train_loader, test_loader, device,
               epochs=2, lr=0.001, label="post-quant")

# ── Save compressed .npz ──────────────────────────────────────────────────────
npz_path = "checkpoints/mnist_compressed.npz"
save_model_npz(model, npz_path)

# ── Reload and verify ─────────────────────────────────────────────────────────
print("\n── [5] Reload & verify ──")
model2 = mnist_model().to(device)
model2 = load_model_from_npz(model2, npz_path, device=str(device))
acc    = evaluate(model2, test_loader, device)
print(f"  Reloaded model accuracy: {acc:.2f}%")
