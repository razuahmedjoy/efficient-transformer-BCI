# Progress Update — EfficientEEGEncoder v2 for BCI Motor Imagery

**Date:** February 2026  
**Dataset:** BCI Competition IV-2a (9 subjects, 4-class motor imagery)  
**Reference paper:** Liao et al., Scientific Reports 2025 — EEGEncoder. Code: `EEGEncoder-main/`. Details: `reference_paper_details.md` (project root).

---

## 1. Current status

We implemented **EfficientEEGEncoder v2**, a purpose-built efficient transformer for the same task and dataset as the reference paper. All comparisons below are **against the reference paper only** (86.46% SD, 74.48% LOSO).

- **Subject-dependent (SD):** 81.25% accuracy, 0.75 kappa  
- **Subject-independent (LOSO):** 69.17% (±10.82%), kappa 0.59  
- **Efficiency:** ~34% fewer parameters, ~1.6× faster inference, lower memory (see Section 3)

---

## 2. Results and comparison

### 2.1 Subject-dependent (SD)

| Subject | Accuracy | Kappa |
|:-------:|:--------:|:-----:|
| S1 | 84.38% | 0.79 |
| S2 | 62.15% | 0.50 |
| S3 | 93.75% | 0.92 |
| S4 | 76.39% | 0.69 |
| S5 | 80.21% | 0.74 |
| S6 | 70.49% | 0.61 |
| S7 | 91.32% | 0.88 |
| S8 | 87.50% | 0.83 |
| S9 | 85.07% | 0.80 |
| **Mean** | **81.25%** | **0.75** |

**Reference paper:** 86.46% acc, 0.82 kappa → **Gap:** −5.21% acc, −0.07 kappa.

### 2.2 Subject-independent (LOSO)

| Subject | Accuracy | Kappa |
|:-------:|:--------:|:-----:|
| S1 | 75.17% | 0.67 |
| S2 | 51.22% | 0.35 |
| S3 | 85.42% | 0.81 |
| S4 | 57.64% | 0.44 |
| S5 | 66.67% | 0.56 |
| S6 | 57.81% | 0.44 |
| S7 | 76.74% | 0.69 |
| S8 | 78.65% | 0.72 |
| S9 | 73.26% | 0.64 |
| **Mean** | **69.17%** | **0.59** |
| **Std** | ±10.82% | |

**Reference paper:** 74.48% LOSO → **Gap:** −5.3%.

LOSO uses Euclidean Alignment (EA), 600 epochs, augmentation, and the same evaluation protocol as the reference.

---

## 3. Efficiency comparison (vs reference)

All metrics: reference implementation (EEGEncoder) vs EfficientEEGEncoder v2, same task, batch 16, AMP where applicable.

| Metric | Reference (EEGEncoder) | Our model (v2) | Change |
|--------|------------------------|----------------|--------|
| **Trainable parameters (SD)** | 181,332 | **119,316** | **−34%** |
| **Inference time (batch 16, AMP)** | ~25 ms | **~15 ms** | **~1.6× faster** |
| **Peak GPU memory — inference** | ~80–100 MB | **~45–65 MB** | **~40–50% less** |
| **Peak GPU memory — training (batch 64)** | Higher | Lower | **~30–50% less** (typical) |
| **Relative inference FLOPs** | 1× | **~0.55–0.65×** | **~35–45% less** |

**Verdict:** The model is **more efficient** than the reference (fewer parameters, less memory, less compute), with an accuracy trade-off of about 5% (SD and LOSO).

---

## 4. What is in this folder

| Item | Description |
|------|-------------|
| **efficient_eegencoder.py** | EfficientEEGEncoder v2: shared transformer, Flash Attention, L2 regularization, no LLM overhead. |
| **train_efficient.py** | SD and LOSO training; EA, augmentation, checkpoints, resume. |
| **train_efficient.ipynb** | Colab notebook for SD (same recipe as script). |
| **subject_independent_eval.ipynb** | LOSO evaluation with EA and config options. |
| **alignment.py** | Euclidean Alignment (used by LOSO when EA is on). |

Results (e.g. `results_sd.json`, `results_loso.json`) are written to the configured results directory (e.g. `results_v2/`).

---

## 5. How to run

- **Subject-dependent:**  
  `python train_efficient.py --mode sd --data_dir /path/to/data/`  
  Or run `train_efficient.ipynb` on Colab.

- **Subject-independent (LOSO):**  
  `python train_efficient.py --mode loso --data_dir /path/to/data/`  
  EA is on by default. For ablation without EA: `--no_ea`.

See **KAGGLE_INSTRUCTIONS.md** (project root) for Kaggle setup and time estimates.

---

## 6. Next steps (short plan)

Possible directions to improve accuracy (especially subject independence) while staying efficient:

1. **Pre-training / more data** — Masked or contrastive pre-training on BCI-IV-2a (+ optional MOABB datasets); or use an external EEG foundation model (e.g. REVE) as a frozen feature extractor. See **experiment_v3/** and **IMPROVEMENT_ROADMAP.md**.
2. **Stronger augmentation** — Mixup, temporal crop, or tuned augmentation for LOSO.
3. **Alignment and domain-invariant training** — Riemannian alignment vs EA; tune subject-adversarial weight; optional CORAL/MMD loss.
4. **Light capacity increase** — Try hidden size 40–48 or 3 transformer layers (stay below reference params).
5. **Training tweaks** — LOSO 700–800 epochs; multi-seed ensemble for more stable metrics.
