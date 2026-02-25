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

 INPUT:  Raw .mat files (A01T.mat, A01E.mat, ..., A09T.mat, A09E.mat)
         These are the BCI Competition IV-2a dataset files.

 OUTPUT: 9 .pkl files (data_all_1.pkl, ..., data_all_9.pkl)
         Each contains: (X_train, X_test, y_train_onehot, y_test_onehot)
         Shape: (288, 1, 22, 1125) for X; (288, 4) for y

 INSTALL: pip install scipy scikit-learn numpy
=============================================================================
"""

import numpy as np          # Array operations (main data structure)
import scipy.io as sio      # Load MATLAB .mat files
from sklearn.preprocessing import StandardScaler  # Z-score normalization
from sklearn.utils import shuffle                # Shuffle data with random state
import pickle               # Save processed data as binary .pkl files
import os                   # File path operations


def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """
    Loading and Dividing of the data set based on the subject-specific
    (subject-dependent) approach.
    
    THIS IS THE EXACT SAME FUNCTION FROM THE ORIGINAL preprocess.py.
    No modifications — ensures results match the reference paper.
    
    The BCI IV-2a dataset contains:
    - 9 subjects performing 4 motor imagery tasks
    - Each subject has 2 sessions: Training (T) and Evaluation (E)
    - Each session has 6 runs of 48 trials = 288 trials total
    - Each trial: 2s fixation + cue + 4.5s motor imagery
    - 22 EEG channels recorded at 250 Hz
    
    Parameters
    ----------
    data_path : str
        Path to the directory containing .mat files.
        Must end with '/'
    subject : int
        Subject number in [1, .., 9]
    training : bool
        If True, load training data (A0xT.mat)
        If False, load testing data (A0xE.mat)
    all_trials : bool
        If True, load all trials (including rejected ones)
        If False, ignore trials with artifacts
    
    Returns
    -------
    data_return : np.array, shape (N_trials, 22, 1125)
        EEG data for each trial (22 channels, 4.5 seconds at 250 Hz)
    class_return : np.array, shape (N_trials,)
        Labels for each trial (0=left hand, 1=right hand, 2=both feet, 3=tongue)
    """
    
    # ---- Define trial parameters ----
    n_channels = 22          # Number of EEG channels in BCI IV-2a
    n_tests = 6 * 48         # 288 trials per session (6 runs × 48 trials/run)
    window_Length = 7 * 250   # 1750 samples = 7 seconds at 250 Hz sampling rate

    # ---- Define the Motor Imagery time window ----
    # We extract a wider window first (7s), then slice to the MI-relevant portion
    fs = 250                 # Sampling rate in Hz
    t1 = int(1.5 * fs)      # Start index = 375 (1.5 seconds after trial onset)
    t2 = int(6 * fs)        # End index = 1500 (6.0 seconds after trial onset)
    # This gives us t2-t1 = 1125 samples = 4.5 seconds of motor imagery data

    # ---- Pre-allocate arrays ----
    class_return = np.zeros(n_tests)                           # Labels
    data_return = np.zeros((n_tests, n_channels, window_Length)) # EEG data

    # ---- Load the .mat file ----
    NO_valid_trial = 0  # Counter for valid (non-artifact) trials
    if training:
        # Load training session file (e.g., A01T.mat)
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        # Load evaluation session file (e.g., A01E.mat)
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    
    # ---- Parse the nested MATLAB structure ----
    # The .mat file has a nested structure:
    # data[0, run] → [0, 0] → [X, trial_onsets, labels, fs, classes, artifacts]
    a_data = a['data']
    for ii in range(0, a_data.size):          # Iterate over runs (6 runs)
        a_data1 = a_data[0, ii]               # Get run ii
        a_data2 = [a_data1[0, 0]]             # Unpack nested structure
        a_data3 = a_data2[0]                  # Get the actual data arrays
        a_X = a_data3[0]                      # EEG signal: (timepoints, channels)
        a_trial = a_data3[1]                  # Trial onset indices
        a_y = a_data3[2]                      # Class labels (1-4)
        a_artifacts = a_data3[5]              # Artifact flags (0=clean, 1=artifact)

        # ---- Extract each trial ----
        for trial in range(0, a_trial.size):   # Iterate over trials in this run
            # Skip artifact trials if all_trials is False
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            
            # Get the starting sample index for this trial
            # Use .item() because some .mat files store indices as nested numpy arrays
            trial_start = int(np.asarray(a_trial[trial]).item())
            
            # Extract 7-second window of 22-channel EEG starting at trial onset
            # a_X shape is (total_samples, channels), we take 22 EEG channels
            # Transpose to get (channels, timepoints) format
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[trial_start:(trial_start + window_Length), :22]
            )
            
            # Store the class label (convert nested array to scalar)
            class_return[NO_valid_trial] = int(np.asarray(a_y[trial]).item())
            NO_valid_trial += 1

    # ---- Slice to Motor Imagery window and fix labels ----
    # Only keep the MI window (1.5s to 6.0s) from the full 7s window
    data_return = data_return[0:NO_valid_trial, :, t1:t2]   # Shape: (288, 22, 1125)
    class_return = class_return[0:NO_valid_trial]
    # Convert labels from 1-indexed (MATLAB) to 0-indexed (Python)
    class_return = (class_return - 1).astype(int)            # [1,2,3,4] → [0,1,2,3]

    return data_return, class_return


def standardize_data(X_train, X_test, channels):
    """
    Z-score standardization per EEG channel — EXACT COPY from original preprocess.py.
    
    For each of the 22 channels:
    1. Fit the scaler on training data (learn mean and std)
    2. Transform both training and test data using those stats
    
    This prevents data leakage: test data is normalized using
    training statistics, not its own statistics.
    
    Parameters
    ----------
    X_train : np.array, shape (N, 1, 22, 1125)
    X_test  : np.array, shape (N, 1, 22, 1125)
    channels : int, number of EEG channels (22)
    """
    for j in range(channels):
        scaler = StandardScaler()
        # Fit scaler on training channel j: learns mean & std across all trials
        scaler.fit(X_train[:, 0, j, :])        # Shape: (N_train, 1125)
        # Transform train and test using the SAME mean & std
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test


def get_data(path, subject, dataset='BCI2a', n_classes=4, isStandard=True, isShuffle=True):
    """
    Complete data loading pipeline for one subject.
    
    Steps:
    1. Load training + evaluation sessions from .mat files
    2. Shuffle the trials (same random state for reproducibility)
    3. Reshape from 3D (N, 22, 1125) to 4D (N, 1, 22, 1125) for Conv2d
    4. One-hot encode labels: [0,1,2,3] → [[1,0,0,0], [0,1,0,0], ...]
    5. Standardize each channel (z-score normalization)
    
    Returns
    -------
    X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
    """
    # Step 1: Load raw data for this subject
    X_train, y_train = load_BCI2a_data(path, subject + 1, True)   # Training session
    X_test, y_test = load_BCI2a_data(path, subject + 1, False)    # Evaluation session

    # Step 2: Shuffle to break any temporal ordering within sessions
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Step 3: Reshape to 4D for Conv2d input
    # From (N, 22, 1125) → (N, 1, 22, 1125)
    # The "1" is the input channel dimension expected by Conv2d
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    
    # Step 4: One-hot encode labels
    # np.eye(4)[label] converts scalar label to one-hot vector
    y_train_onehot = np.eye(n_classes)[y_train]

    # Same for test data
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = np.eye(n_classes)[y_test]

    # Step 5: Standardize each channel using training statistics
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


def data_save(data_path, output_dir, n_sub=9, n_classes=4):
    """
    Main preprocessing entry point — processes all 9 subjects.
    
    For each subject:
    1. Calls get_data() to load, shuffle, reshape, and standardize
    2. Packs as tuple: (X_train, X_test, y_train_onehot, y_test_onehot)
    3. Saves to .pkl file: data_all_{subject}.pkl
    
    The .pkl format matches what EEGDB class in training scripts expects.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for sub in range(n_sub):
        print(f"\n--- Subject {sub + 1} ---")

        # Load and preprocess all data for this subject
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, n_classes=n_classes, isStandard=True)

        # Pack into tuple (this exact format is expected by EEGDB DataLoader)
        data_to_save = (X_train, X_test, y_train_onehot, y_test_onehot)
        save_path = os.path.join(output_dir, f'data_all_{sub + 1}.pkl')

        # Save as pickle binary file
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
        print(f"  ✅ Saved: {save_path}")


# ==================== ENTRY POINT ====================
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

    # Run the full preprocessing pipeline
    data_save(data_path, output_dir)
    print("\n✅ Preprocessing complete! PKL files ready for training.")
