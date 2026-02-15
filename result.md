# EEGEncoder Baseline Reproduction Results

**Training:** 500 epochs, batch size 64, Adam (lr=1e-3), CrossEntropy (label smoothing=0.2)

## Subject-Dependent Results

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

## Comparison with Paper

| Metric   | Paper    | Ours     | Difference |
|:---------|:--------:|:--------:|:----------:|
| Accuracy | 86.46%   | 84.72%   | -1.74%     |
| Kappa    | 0.82     | 0.80     | -0.02      |

> Results are close to the paper's reported values (~1.7% difference), which is within
> normal variation for different hardware/software environments.