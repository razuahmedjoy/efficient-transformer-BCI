"""
=============================================================================
 EEGEncoder - Colab Data Preprocessing (Using .mat files — same as original)
=============================================================================
 PURPOSE: This is the original preprocess.py adapted for Google Colab.
          Uses the EXACT SAME loading logic from the reference paper code.
          The only change is making paths configurable.

 ORIGINAL CODE: EEGEncoder-main/preprocess.py
 CHANGES:       - Paths are configurable (not hardcoded)
                - Added print statements for progress tracking
                - No changes to data loading, slicing, or standardization

 INSTALL: pip install scipy scikit-learn numpy
=============================================================================
"""

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle
import os


def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """
    Loading and Dividing of the data set based on the subject-specific
    (subject-dependent) approach.
    
    THIS IS THE EXACT SAME FUNCTION FROM THE ORIGINAL preprocess.py.
    No modifications — ensures results match the reference paper.
    
    Parameters
    ----------
    data_path : str
        Path to the directory containing .mat files.
        Must end with '/' and contain subject subdirectories like s1/, s2/, etc.
        OR contain files directly (see get_data for path construction).
    subject : int
        Subject number in [1, .., 9]
    training : bool
        If True, load training data (A0xT.mat)
        If False, load testing data (A0xE.mat)
    all_trials : bool
        If True, load all trials (including rejected ones)
        If False, ignore trials with artifacts
    """
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6 * 48     # 288 trials per session
    window_Length = 7 * 250  # 1750 samples (7 seconds at 250 Hz)

    # Define MI trial window
    fs = 250            # sampling rate
    t1 = int(1.5 * fs)  # 375 — start at 1.5s after trial onset
    t2 = int(6 * fs)    # 1500 — end at 6.0s after trial onset

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            # Use .item() to safely extract scalar from numpy array
            trial_start = int(np.asarray(a_trial[trial]).item())
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[trial_start:(trial_start + window_Length), :22]
            )
            class_return[NO_valid_trial] = int(np.asarray(a_y[trial]).item())
            NO_valid_trial += 1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return


def standardize_data(X_train, X_test, channels):
    """
    StandardScaler per channel — EXACT COPY from original preprocess.py.
    Fit on training data, transform both train and test.
    """
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test


def get_data(path, subject, dataset='BCI2a', n_classes=4, isStandard=True, isShuffle=True):
    """
    Load, split, reshape, standardize — EXACT COPY from original preprocess.py.
    Only change: path construction uses direct path instead of s{subject}/ subdir.
    """
    X_train, y_train = load_BCI2a_data(path, subject + 1, True)
    X_test, y_test = load_BCI2a_data(path, subject + 1, False)

    # Shuffle the data
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Prepare training data
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = np.eye(n_classes)[y_train]

    # Prepare testing data
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = np.eye(n_classes)[y_test]

    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


def data_save(data_path, output_dir, n_sub=9, n_classes=4):
    """
    Main preprocessing function — matches original data_save().
    
    Parameters
    ----------
    data_path : str
        Directory containing .mat files (A01T.mat, A01E.mat, ...).
        Must end with '/'.
    output_dir : str
        Directory to save .pkl files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for sub in range(n_sub):
        print(f"\n--- Subject {sub + 1} ---")

        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, n_classes=n_classes, isStandard=True)

        data_to_save = (X_train, X_test, y_train_onehot, y_test_onehot)
        save_path = os.path.join(output_dir, f'data_all_{sub + 1}.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
        print(f"  ✅ Saved: {save_path}")


if __name__ == '__main__':
    # === CONFIGURE THESE PATHS ===
    # Point data_path to the directory containing A01T.mat, A01E.mat, etc.
    # MUST end with '/'

    # For Google Colab with Drive:
    data_path  = 'EEGEncoder-main/datasets/'
    output_dir = 'EEGEncoder-main/data/'

    # For local:
    # data_path  = './EEGEncoder-main/datasets/'
    # output_dir = './EEGEncoder-main/data/'

    data_save(data_path, output_dir)
    print("\n✅ Preprocessing complete! PKL files ready for training.")
