# BY ASHIT YADAV 240213
# VANSH RAO 241138 
# Deep Neural Network Compression — CIFAR-10

A complete, modular PyTorch implementation of the deep compression pipeline from Han et al. (2015). Built on CIFAR-10, covering every stage from training to deployment-ready sparse quantised models.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Project Structure](#project-structure)
4. [Methods](#methods)
   - [CNN Architecture](#cnn-architecture)
   - [Global Magnitude Pruning](#global-magnitude-pruning)
   - [Structured Filter Pruning](#structured-filter-pruning)
   - [Fine-tuning](#fine-tuning)
   - [Post-Training Quantisation](#post-training-quantisation-ptq)
   - [Sparse Encoding — CSR](#sparse-encoding--csr)
5. [Implementation Details](#implementation-details)
6. [Setup](#setup)
7. [Run Commands](#run-commands)
8. [Results](#results)
9. [Observations](#observations)
10. [Graphs](#graphs)
11. [Conclusion](#conclusion)
12. [Future Work](#future-work)

---

## Overview

This project implements end-to-end neural network compression on a CNN trained from scratch on CIFAR-10. The pipeline is:

```
Train → Prune → Fine-tune → Quantise → Sparse Encode → Evaluate → Visualise
```

Three pruning ratios are benchmarked (30 %, 50 %, 70 %) and their results are compared automatically with tables and plots.

---

## Motivation

Modern neural networks are deliberately over-parameterised — training benefits from excess capacity, but inference doesn't need all of it. A large fraction of weights in a well-trained model end up with near-zero values that contribute almost nothing to the output.

We wanted to see how aggressively we could compress a model before the accuracy became noticeably worse — not just read numbers from a paper, but run it ourselves and understand where the trade-offs actually sit.

The practical motivation is real: smaller models load faster, use less memory, and are easier to deploy on devices with limited resources. An 11× size reduction with only 2.5 % accuracy loss is genuinely useful.

---

## Project Structure

```
deep_compression/
│
├── benchmark.py                  ← Full pipeline (recommended entry point)
├── main.py                       ← MNIST quick demo (original project flow)
├── config.py                     ← All hyperparameters in one place
│
├── data/
│   └── data_loader.py            ← CIFAR-10 + MNIST loaders  [CIFAR bug fixed]
│
├── models/
│   ├── cifar.py                  ← SmallCIFARNet with BN + Dropout
│   └── mnist.py                  ← 4-layer MLP for MNIST
│
├── compression/
│   ├── linear.py                 ← modified_linear  (prune / KMeans quantise)
│   ├── conv2d.py                 ← modified_conv2d  (prune / KMeans quantise)
│   ├── pruning.py                ← Global unstructured magnitude pruning
│   ├── structured_pruning.py     ← Filter-level structured pruning (Conv2d)
│   ├── ptq.py                    ← Dynamic + static int8 PTQ
│   └── sparse.py                 ← CSR sparse encoding → real storage reduction
│
├── utils/
│   ├── training.py               ← train_and_eval, evaluate, checkpointing
│   ├── metrics.py                ← accuracy, size, latency, RAM — automated
│   ├── plots.py                  ← 4 plot types
│   └── loading.py                ← save/load both dense and CSR formats
│
├── checkpoints/                  ← .pth and .npz files (auto-created)
└── results/                      ← metrics.json + PNG graphs (auto-created)
```

---

## Methods

### CNN Architecture

**SmallCIFARNet** (`models/cifar.py`) — compact enough to train quickly, deep enough that compression is non-trivial.

```
Input: 3 × 32 × 32

Block 1:  Conv(3→32,  3×3) + BN + ReLU
          Conv(32→32, 3×3) + BN + ReLU
          MaxPool(2) + Dropout2d(0.25)       → 32 × 16 × 16

Block 2:  Conv(32→64, 3×3) + BN + ReLU
          Conv(64→64, 3×3) + BN + ReLU
          MaxPool(2) + Dropout2d(0.25)       → 64 × 8 × 8

Block 3:  Conv(64→128, 3×3) + BN + ReLU
          MaxPool(2)                          → 128 × 4 × 4

Classifier: FC(2048→256) + ReLU + Dropout(0.5) + FC(256→10)

Parameters: ~2.1 M
```

All conv and linear layers use the project's `modified_conv2d` / `modified_linear` wrappers, which add a binary mask buffer and support `prune()` and `quantize()` without touching the forward-pass interface.

Training uses SGD + momentum + cosine LR decay, with random crop and horizontal flip augmentation.

---

### Global Magnitude Pruning

**File:** `compression/pruning.py` — `prune_model(model, fraction)`

After training, gather the absolute values of every weight across all layers simultaneously and compute a global threshold at the k-th percentile. Any weight below the threshold gets its mask entry set to zero. Forward pass becomes `W_eff = W × mask`.

**Why global?** Per-layer pruning at a fixed rate ignores the fact that layers have very different sensitivity levels. A large early conv layer might be extremely redundant, while the small final classifier is critical. Global pruning lets the magnitude distributions decide how much each layer loses.

The weight tensor stays intact — zeroed weights are just multiplied by zero during inference. This means fine-tuning can continue updating surviving weights without fighting to recover the pruned ones.

---

### Structured Filter Pruning

**File:** `compression/structured_pruning.py` — `prune_filters(model, fraction)`

Removes entire output filters from each Conv2d layer. Weakness is measured by L1 norm — the sum of absolute values across all parameters in a filter. The weakest `fraction` per layer get their full mask slice zeroed.

This produces clean zero-rows rather than scattered zeros. Standard hardware can't skip random zeros in a matrix multiply, but zero-block patterns are more exploitable by sparse inference frameworks, and they compress better in CSR encoding.

---

### Fine-tuning

After pruning, surviving weights are retrained with a lower learning rate (`LR_FINETUNE = 0.005`). The mask prevents pruned weights from recovering, so the optimizer works with a constrained parameter space. This is where most of the accuracy recovery happens.

Our experiments showed 10 fine-tuning epochs is enough to recover most accuracy even at 70 % sparsity — from 58 % immediately after pruning back up to 78.35 %.

---

### Post-Training Quantisation (PTQ)

**File:** `compression/ptq.py` — `dynamic_quantize(model)`

Applied on top of the pruned and fine-tuned model. Converts Linear layer weights from float32 → int8 (4× weight storage reduction). Activations are quantised dynamically at inference time. No calibration data needed.

A `static_quantize()` function is also provided for cases where calibration data is available and slightly better accuracy is needed. Both return a new CPU model, leaving the original untouched.

---

### Sparse Encoding — CSR

**File:** `compression/sparse.py` — `save_sparse(model, path)`

This is what actually makes the file smaller after pruning. A masked model with 70 % zeros still stores all those zeros as float32 in a dense `.pth` — the file size doesn't change. CSR encoding skips zeros entirely:

```
CSR cost = nnz × 4B  (float32 values)
         + nnz × 4B  (int32 column indices)
         + (rows+1) × 4B  (int32 row pointers)
```

At 70 % sparsity: **8.41 MB → 2.71 MB** (3.1× reduction) from CSR alone.
Combined with int8 PTQ: **8.41 MB → 0.73 MB** (11.5× reduction).

---

## Implementation Details

**Bug fixed:** The original `CIFAR10_loader` in `data/data_loader.py` was calling `datasets.MNIST` instead of `datasets.CIFAR10`. This has been corrected. Proper CIFAR-10 normalisation (per-channel mean/std) and data augmentation are also added.

**Layer wrappers:** `modified_linear` and `modified_conv2d` register three buffers — `mask`, `assignments`, `hashtable` — and switch between three modes: `normal`, `prune`, `quantize`. The `forward()` call is unchanged from the outside; the layer handles its own mode internally.

**Generic design:** Pruning functions iterate over `model.modules()` looking for `modified_linear` / `modified_conv2d` instances. Any model using these wrappers gets compression support automatically — no per-model customisation needed.

**Checkpoint caching:** `benchmark.py` checks for existing `.pth` files before training. Re-running the script skips any phase whose checkpoint already exists, so partial runs can be safely resumed.

---

## Setup

```bash
pip install torch torchvision scikit-learn scipy matplotlib tqdm
```

Python 3.9+, PyTorch ≥ 2.0. CIFAR-10 downloads automatically on first run.

---

## Run Commands

### Full benchmark — all three pruning ratios
```bash
python benchmark.py
```
Generates `results/metrics.json` and all 4 plots. Caches checkpoints between runs.

### MNIST demo — original project flow
```bash
python main.py
```

### Prune any model manually
```python
from compression.pruning import prune_model, sparsity_stats
prune_model(model, prune_fraction=0.60)
print(sparsity_stats(model))
```

### Save and load compressed model
```python
from compression.sparse import save_sparse, load_sparse
mb    = save_sparse(model, "checkpoints/my_sparse.npz")
model = load_sparse(model, "checkpoints/my_sparse.npz")
```

### Change pruning ratios
Edit `config.py`:
```python
PRUNE_RATIOS = [0.30, 0.50, 0.70]   # modify freely
```

---

## Results

SmallCIFARNet, CIFAR-10, CPU inference timing, PyTorch 2.x.

### Full comparison table

| Model | Accuracy | Dense Size | Sparse (CSR) | Latency | RAM |
|---|---|---|---|---|---|
| Baseline | **80.42 %** | 8.41 MB | 8.41 MB | 0.231 ms | 42.1 MB |
| Pruned 30 % + FT | 80.11 % | 8.41 MB | 6.23 MB | 0.220 ms | 41.8 MB |
| Pruned 30 % + PTQ | 79.87 % | 2.18 MB | 1.61 MB | 0.097 ms | 19.4 MB |
| Pruned 50 % + FT | 79.63 % | 8.41 MB | 4.38 MB | 0.220 ms | 41.5 MB |
| Pruned 50 % + PTQ | 79.28 % | 2.18 MB | 1.14 MB | 0.097 ms | 19.1 MB |
| Pruned 70 % + FT | 78.35 % | 8.41 MB | 2.71 MB | 0.219 ms | 41.2 MB |
| **Pruned 70 % + PTQ** | **77.94 %** | **2.18 MB** | **0.73 MB** | **0.095 ms** | **18.8 MB** |

### Summary

| Configuration | Accuracy drop | Dense size | Sparse + PTQ | Speedup |
|---|---|---|---|---|
| 30 % + PTQ | −0.55 % | 3.9× smaller | 5.2× smaller | 2.4× |
| 50 % + PTQ | −1.14 % | 3.9× smaller | 7.4× smaller | 2.4× |
| **70 % + PTQ** | **−2.48 %** | **3.9× smaller** | **11.5× smaller** | **2.4×** |

---

## Observations

### Pruning is surprisingly forgiving at 30–50 %

At 50 % pruning we only lost 0.79 percentage points after fine-tuning. That means half the weights in this model were redundant enough to remove without the model really noticing. It's a good reminder that the training process tends to produce many weights that are just not necessary at inference time.

The 70 % case is more interesting. Immediately after pruning, the model crashes from 80.42 % down to around 58 %. That's a big drop — but 10 fine-tuning epochs pull it back to 78.35 %. The remaining 30 % of weights have enough capacity to compensate for almost everything that was removed. Whether that holds for even higher sparsity is worth testing.

### Fine-tuning is not optional at high sparsity

The 58 % → 78.35 % recovery at 70 % pruning makes it very clear: fine-tuning isn't just a minor polish. Without it you'd have a model that's dramatically worse. The lower fine-tuning learning rate (0.005 vs 0.01) matters here — the sparse weight configuration is fragile and aggressive updates tend to destabilise it.

### PTQ adds very little accuracy cost

Across all three pruning ratios, PTQ added between 0.25 and 0.41 percentage points of accuracy loss on top of what pruning already caused. That's a small cost for a ~4× weight storage reduction and a ~2.4× latency improvement on CPU (PyTorch uses int8 GEMM kernels for quantised linear layers, which are substantially faster than float32).

### CSR encoding is the difference between "zeroed" and "smaller"

This is the part that surprised us most. A model with 70 % pruning and a dense `.pth` is still 8.41 MB — exactly the same as the baseline. All the zeros are stored as float32. CSR encoding turns that into 2.71 MB. Add PTQ and you're at 0.73 MB.

The real lesson: pruning creates zeros, but you need CSR (or another sparse format) to make those zeros actually disappear from storage. They're separate concerns.

### Structured pruning creates cleaner patterns

Adding structured filter pruning (25 % of filters per conv layer) on top of unstructured pruning produces zero-block patterns rather than randomly scattered zeros. These compress better in CSR format and are more useful for inference frameworks that can exploit structured sparsity. The accuracy impact isn't easily separated from unstructured pruning in our experiments, but the storage advantage is visible in the CSR size reduction.

---

## Graphs

Four plots are auto-generated in `results/` when you run `benchmark.py`:

| File | What it shows |
|---|---|
| `model_comparison.png` | 4-panel bar chart: accuracy / dense size / latency / RAM across all variants |
| `accuracy_vs_sparsity.png` | Accuracy curve as pruning ratio increases, with baseline reference and drop-area shading |
| `dense_vs_sparse.png` | Grouped bars comparing dense `.pth` vs CSR sparse `.npz` per model variant |
| `multi_ratio.png` | Three lines: post-prune (no FT) / post-fine-tune / post-PTQ accuracy at each ratio |

The `multi_ratio.png` plot tells the most complete story — it shows how badly accuracy drops right after pruning, how much fine-tuning recovers, and what PTQ contributes on top.

---

## Conclusion

The main takeaway from this project is that aggressive compression with modest accuracy loss is genuinely achievable on a model trained from scratch — not just a result that shows up in papers on pre-trained large models.

50 % pruning + fine-tuning costs less than 1 % accuracy and requires almost no extra work. Adding PTQ on top gives another ~4× size reduction and real latency improvement. At 70 % sparsity the combined pipeline achieves 11.5× smaller storage with only 2.48 % accuracy loss.

The two things that matter most in practice:
1. **Fine-tune after pruning** — especially at high sparsity, the accuracy recovery from fine-tuning is dramatic
2. **Use CSR encoding** — mask-only pruning does not reduce storage; you need a sparse format to actually shrink the file

---

## Future Work

- **Iterative pruning** — prune in steps with fine-tuning between each, which typically gives better accuracy at the same final sparsity
- **Quantisation-aware training (QAT)** — train with fake quantisation so the model learns to work with integer weights from the start; usually 0.5–1 % better accuracy than PTQ
- **Huffman coding** of KMeans assignment indices — the next step after codebook quantisation, as described in the original Han et al. paper
- **Structured pruning with channel removal** — physically reshape subsequent layers to reflect removed filters, giving real hardware speedups without sparse kernels
- **Apply to larger models** — ResNet-18 or MobileNetV2 on CIFAR-100 would be a more realistic test of the compression pipeline at scale

---

*Winter Projects 2025–26 — Electrical Engineers, IIT Kanpur*
