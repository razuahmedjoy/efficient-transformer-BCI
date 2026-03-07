"""
Euclidean Alignment (EA) for cross-subject EEG normalization.

EA aligns each subject's covariance structure to a shared reference,
removing the bulk of subject-specific distributional shift. This is the
single most impactful preprocessing step for subject-independent EEG-MI.

Reference:
    He & Wu, "Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach", IEEE TBME 2020.
"""

import numpy as np
from scipy.linalg import sqrtm, inv


def _real(M):
    """Ensure matrix is real-valued (sqrtm can return complex with tiny imag)."""
    return np.real(M).astype(np.float32)


def compute_covariance(X):
    """Average trial covariance.

    Args:
        X: (N, C, T) — N trials, C channels, T timepoints.
    Returns:
        R: (C, C) mean covariance matrix.
    """
    N, C, T = X.shape
    R = np.zeros((C, C), dtype=np.float64)
    for i in range(N):
        R += X[i] @ X[i].T / T
    R /= N
    R += 1e-6 * np.eye(C)
    return R


def euclidean_alignment(X, R_ref=None):
    """Apply EA to a single subject's data. X: (N, C, T). Returns aligned (N, C, T)."""
    R = compute_covariance(X)
    R_inv_sqrt = _real(inv(sqrtm(R)))
    if R_ref is not None:
        R_ref_sqrt = _real(sqrtm(R_ref))
        transform = R_ref_sqrt @ R_inv_sqrt
    else:
        transform = R_inv_sqrt
    X_aligned = np.einsum("ij,njt->nit", transform, X)
    return X_aligned, R


def align_all_subjects(subjects_data):
    """Align list of (N_s, C, T) arrays. Returns aligned list, R_ref."""
    covs = [compute_covariance(X) for X in subjects_data]
    R_ref = np.mean(covs, axis=0)
    aligned = []
    for X in subjects_data:
        X_a, _ = euclidean_alignment(X, R_ref)
        aligned.append(X_a)
    return aligned, R_ref
