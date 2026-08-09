"""
benchmark.py
────────────
Complete DNN compression pipeline for CIFAR-10.

Pipeline
─────────
  Train baseline SmallCIFARNet  →  measure
  For each ratio in PRUNE_RATIOS [30%, 50%, 70%]:
    ├── Global magnitude pruning
    ├── Structured filter pruning  (optional)
    ├── Save sparse CSR .npz       (actual storage reduction)
    ├── Fine-tune
    ├── Measure  (accuracy, size, latency, RAM)
    └── Dynamic int8 PTQ  →  measure again
  ──────────────────────────────────────────
  Print full comparison table
  Save results/metrics.json
  Generate 4 PNG graphs

Run
────
    python benchmark.py

Checkpoints are cached — re-running loads from disk, skips training.
"""

import os
import json
import copy
import torch

from config import (
    get_device,
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, AUGMENT,
    EPOCHS_BASELINE, EPOCHS_FINETUNE,
    LR_BASELINE, LR_FINETUNE, MOMENTUM, WEIGHT_DECAY, LR_SCHEDULER,
    PRUNE_RATIOS,
    STRUCTURED_PRUNE, STRUCTURED_PRUNE_RATIO,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from data.data_loader               import CIFAR10_loader
from models.cifar                   import SmallCIFARNet
from compression.pruning            import prune_model, sparsity_stats
from compression.structured_pruning import prune_filters, filter_sparsity
from compression.ptq                import dynamic_quantize
from compression.sparse             import save_sparse
from utils.training                 import train_and_eval, save_checkpoint, load_checkpoint, evaluate
from utils.metrics                  import full_metrics, print_table
from utils.plots                    import (
    plot_model_comparison,
    plot_accuracy_vs_sparsity,
    plot_dense_vs_sparse,
    plot_multi_ratio,
)

# ─────────────────────────────────────────────────────────────────────────────
device = get_device()

print("\n" + "═" * 60)
print("  Deep Neural Network Compression Benchmark  —  CIFAR-10")
print("═" * 60 + "\n")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("─" * 60)
print("  [1]  CIFAR-10 data")
print("─" * 60)
train_loader, test_loader = CIFAR10_loader(
    path=DATA_DIR, batch_size=BATCH_SIZE,
    augment=AUGMENT, num_workers=NUM_WORKERS,
)

# ── 2. Baseline Training ──────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  [2]  Baseline SmallCIFARNet")
print("─" * 60)

baseline      = SmallCIFARNet().to(device)
baseline_ckpt = os.path.join(CHECKPOINT_DIR, "baseline.pth")

if os.path.exists(baseline_ckpt):
    print("  Found existing baseline checkpoint — loading.")
    baseline, _, baseline_acc = load_checkpoint(baseline, baseline_ckpt, device)
else:
    baseline_acc = train_and_eval(
        baseline, train_loader, test_loader, device,
        epochs=EPOCHS_BASELINE, lr=LR_BASELINE,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        scheduler=LR_SCHEDULER, label="baseline",
    )
    save_checkpoint(baseline, baseline_ckpt, epoch=EPOCHS_BASELINE, acc=baseline_acc)

print(f"\n  ✓  Baseline accuracy: {baseline_acc:.2f}%")

# ── 3. Baseline Metrics ───────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  [3]  Baseline metrics")
print("─" * 60)
base_metrics = full_metrics(baseline, test_loader, device, label="Baseline")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Multi-ratio compression loop
# ─────────────────────────────────────────────────────────────────────────────
all_results = [base_metrics]
ratio_curve = []

for ratio in PRUNE_RATIOS:
    pct = int(ratio * 100)
    print(f"\n{'═' * 60}")
    print(f"  Pruning Ratio: {pct}%")
    print(f"{'═' * 60}")

    model = copy.deepcopy(baseline)

    # ── 4a. Global magnitude pruning ──────────────────────────────────────────
    print(f"\n  [4a] Global magnitude pruning @ {pct}%")
    prune_model(model, ratio)
    sp = sparsity_stats(model)
    print(f"       Weight sparsity: {sp['sparsity_pct']:.1f}%")

    # ── 4b. Structured filter pruning (optional) ───────────────────────────────
    if STRUCTURED_PRUNE:
        print(f"\n  [4b] Structured filter pruning ({STRUCTURED_PRUNE_RATIO:.0%})")
        prune_filters(model, fraction=STRUCTURED_PRUNE_RATIO)
        fs = filter_sparsity(model)
        print(f"       Filter sparsity: {fs['filter_sparsity_pct']:.1f}%  "
              f"({fs['zeroed_filters']}/{fs['total_filters']} filters zeroed)")

    acc_post_prune = evaluate(model, test_loader, device)
    print(f"\n  Accuracy right after pruning (before fine-tune): {acc_post_prune:.2f}%")

    # ── 4c. CSR sparse save ────────────────────────────────────────────────────
    sparse_path = os.path.join(CHECKPOINT_DIR, f"sparse_{pct}.npz")
    sparse_mb   = save_sparse(model, sparse_path)

    # ── 4d. Fine-tune ──────────────────────────────────────────────────────────
    print(f"\n  [4d] Fine-tuning ({EPOCHS_FINETUNE} epochs)")
    ft_ckpt = os.path.join(CHECKPOINT_DIR, f"pruned_{pct}_finetuned.pth")

    if os.path.exists(ft_ckpt):
        print("       Found fine-tune checkpoint — loading.")
        model, _, ft_acc = load_checkpoint(model, ft_ckpt, device)
    else:
        ft_acc = train_and_eval(
            model, train_loader, test_loader, device,
            epochs=EPOCHS_FINETUNE, lr=LR_FINETUNE,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
            scheduler=LR_SCHEDULER, label=f"finetune-{pct}%",
        )
        save_checkpoint(model, ft_ckpt, epoch=EPOCHS_FINETUNE, acc=ft_acc)

    print(f"  ✓  Accuracy after fine-tune: {ft_acc:.2f}%")

    # ── 4e. Pruned + FT metrics ────────────────────────────────────────────────
    pruned_metrics = full_metrics(
        model, test_loader, device,
        label=f"Pruned {pct}%+FT",
        sparse_mb=sparse_mb,
    )
    all_results.append(pruned_metrics)

    # ── 4f. Dynamic int8 PTQ ───────────────────────────────────────────────────
    print(f"\n  [4f] Dynamic int8 PTQ on pruned {pct}% model")
    ptq_model  = dynamic_quantize(model)
    cpu_device = torch.device("cpu")

    ptq_metrics = full_metrics(
        ptq_model, test_loader, cpu_device,
        label=f"Pruned {pct}%+PTQ",
        sparse_mb=sparse_mb,
    )
    all_results.append(ptq_metrics)

    ratio_curve.append({
        "ratio":              ratio,
        "acc_after_prune":    round(acc_post_prune, 2),
        "acc_after_finetune": ft_acc,
        "acc_ptq":            ptq_metrics["accuracy"],
    })

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "═" * 60)
print("  COMPLETE RESULTS TABLE")
print("═" * 60)
print_table(all_results)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save JSON
# ─────────────────────────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(json_path, "w") as f:
    json.dump({
        "baseline_acc": base_metrics["accuracy"],
        "results":      all_results,
        "ratio_curve":  ratio_curve,
    }, f, indent=2)
print(f"\n  [saved] {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating plots ...")

sp_ratios = [0.0] + PRUNE_RATIOS
sp_labels = ["Baseline"] + [f"{int(r*100)}%+FT" for r in PRUNE_RATIOS]
sp_accs   = ([base_metrics["accuracy"]]
             + [r["acc_after_finetune"] for r in ratio_curve])

plot_model_comparison(all_results,                       save_dir=RESULTS_DIR)
plot_accuracy_vs_sparsity(sp_ratios, sp_accs, sp_labels, save_dir=RESULTS_DIR)
plot_dense_vs_sparse(all_results,                        save_dir=RESULTS_DIR)
plot_multi_ratio(ratio_curve, base_metrics["accuracy"],  save_dir=RESULTS_DIR)

print(f"\n  Done!  Results → {RESULTS_DIR}/")
print("─" * 60)
