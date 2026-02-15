# Progress Update — EEGEncoder: Efficient Transformer for BCI

**Date:** February 15, 2026  
**Subject:** Progress Report on EEG-Based BCI Classification using Efficient Transformers

---

## 1. Work Completed

### ✅ 1.1 Baseline Reproduction (EEGEncoder Paper — Subject-Dependent)

I successfully reproduced the baseline EEGEncoder model results on the BCI Competition IV-2a dataset (9 subjects, 4-class motor imagery classification) using Google Colab's T4 GPU.

| Subject | Accuracy | Kappa  |
|:-------:|:--------:|:------:|
| S1      | 88.19%   | 0.8426 |
| S2      | 64.58%   | 0.5278 |
| S3      | 94.79%   | 0.9306 |
| S4      | 82.64%   | 0.7685 |
| S5      | 84.38%   | 0.7917 |
| S6      | 76.39%   | 0.6852 |
| S7      | 95.14%   | 0.9352 |
| S8      | 86.46%   | 0.8194 |
| S9      | 89.93%   | 0.8657 |
| **Mean** | **84.72%** | **0.7963** |

**Paper reports:** 86.46% accuracy, 0.82 kappa  
**My reproduction:** 84.72% accuracy, 0.80 kappa  
**Gap:** ~1.74% 

---

### ✅ 1.2 Efficient Model — First Attempt (Subject-Dependent)

I implemented an efficient variant of EEGEncoder with 5 architectural modifications and tested it under the same subject-dependent setting:

| Modification | What Changed | Rationale |
|:--|:--|:--|
| Linear Attention | O(n) instead of O(n²) complexity | Reduce compute for long sequences |
| Reduced Branches | 5 → 3 parallel branches | Fewer parameters (~40% reduction) |
| Depthwise Separable Conv | Lighter convolutions | Reduced conv parameters |
| Shared Transformer | 1 shared vs 5 separate transformers | Major parameter reduction |
| Removed Rotary Embed & Causal Mask | Not needed for EEG classification | Cleaner, faster architecture |

**Results — Efficient Model vs Baseline (Subject-Dependent):**

| Subject | Baseline | Efficient | Difference |
|:-------:|:--------:|:---------:|:----------:|
| S1 | 88.19% | 76.39% | -11.80% |
| S2 | 64.58% | 55.21% | -9.37%  |
| S3 | 94.79% | 87.50% | -7.29%  |
| S4 | 82.64% | 56.94% | -25.70% |
| S5 | 84.38% | 73.96% | -10.42% |
| S6 | 76.39% | 60.07% | -16.32% |
| S7 | 95.14% | 79.17% | -15.97% |
| S8 | 86.46% | 80.21% | -6.25%  |
| S9 | 89.93% | 82.29% | -7.64%  |
| **Mean** | **84.72%** | **72.42%** | **-12.30%** |

The efficient model achieved 72.42% — approximately 12% lower than the baseline.

---

### ⏳ 1.3 Subject-Independent Evaluation (LOSO — Attempted)

I prepared the Leave-One-Subject-Out (LOSO) cross-validation pipeline. LOSO trains on 8 subjects and evaluates on the held-out 9th subject, repeated for all 9 subjects. This tests how well the model generalizes across different individuals — a key requirement for practical BCI systems.

**Status:** I was unable to complete the LOSO evaluation due to Google Colab's free-tier GPU limitations. LOSO requires training 9 separate models (one per fold), each for 200 epochs. After running the baseline and efficient model experiments, the free T4 GPU became unavailable and Colab prompted for a premium subscription. The code and pipeline are fully ready and I need to find any other approach to run this experiment.

---


## 2. Next Steps — My Plan

### 2.1 Systematic Ablation Study

Test one modification at a time against the baseline to isolate each change's effect:

| Experiment | Change Applied | Everything Else |
|:--|:--|:--|
| Ablation 1 | Linear attention only | 5 branches, separate transformers, standard conv |
| Ablation 2 | 3 branches only | Full attention, separate transformers, standard conv |
| Ablation 3 | Shared transformer only | 5 branches, full attention, standard conv |
| Ablation 4 | Depthwise separable conv only | 5 branches, full attention, separate transformers |
| **Final** | **Combine only the beneficial changes** | — |

This will identify which modifications maintain accuracy while reducing compute.

### 2.2 Strategies to Improve Efficient Model Accuracy

- I am still learning transformmer architecture and how efficiency can be improved. I will try to implement some other techniques to improve the efficient model accuracy.

### 2.3 Subject-Independent Evaluation

Once the best efficient model is found:
1. Run LOSO on baseline → establish cross-subject baseline accuracy
2. Run LOSO on best efficient variant → compare generalization
3. If generalization is poor, explore domain adaptation techniques (subject normalization, adversarial training)

### 2.4 Hyperparameter Search & Cross-Dataset Validation

- **Hyperparameter search** for the efficient model (number of attention heads, transformer layers, branches, learning rate)
- **Cross-dataset validation** — test on BCI Competition IV-2b or other motor imagery datasets to verify generalizability


---

## 3. Challenges & Resource Constraints

**Google Colab Free-Tier GPU Limitation:**
- The free T4 GPU becomes unavailable after 2-3 training runs and requires Colab Pro (premium subscription)
- Each full subject-dependent experiment (9 subjects × 500 epochs) takes approximately 2-3 hours
- LOSO evaluation requires 9 separate model trainings, making it even more resource-intensive
- This limits me to about 1-2 experiments per day, significantly slowing the ablation study

**Possible solutions I am exploring:**
- Kaggle Notebooks (alternative free GPU, ~30 hours/week)
- Running smaller experiments on CPU locally (very slow but functional)
- University GPU access if available

---
