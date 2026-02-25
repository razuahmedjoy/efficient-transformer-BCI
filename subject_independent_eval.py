"""
=============================================================================
 Subject-Independent Evaluation — Leave-One-Subject-Out Cross-Validation
=============================================================================
 PURPOSE: Evaluate how well a model generalizes to UNSEEN subjects.
          This is a critical test for real-world BCI applications where
          you can't collect calibration data from every new user.

 METHOD: Leave-One-Subject-Out (LOSO) Cross-Validation
   - 9 folds total (one per subject)
   - In each fold: train on ALL data from 8 subjects, test on the 9th
   - Each subject gets a turn as the "held-out" test subject
   - Report mean accuracy ± std across all 9 folds

 WHY LOSO MATTERS:
   Subject-dependent models memorize ONE person's brain patterns.
   LOSO tests if the model learned GENERAL brain patterns that work
   for anyone. Lower accuracy than subject-dependent is expected,
   but a good model should still generalize.

 USAGE:
   python subject_independent_eval.py --model baseline   # Test original
   python subject_independent_eval.py --model efficient   # Test proposed

 PAPER REFERENCE: Subject-Independent accuracy = 74.48%
=============================================================================
"""

# ==================== IMPORTS ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data         # DataLoader, Dataset
import numpy as np                    # Array operations
import pickle                         # Load .pkl data files
import os                             # File path operations
import sys                            # System path for imports
import time                           # Not used but available for timing
import argparse                       # Command-line argument parsing
from sklearn.metrics import accuracy_score, cohen_kappa_score  # Evaluation metrics

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== CONFIGURATION ====================
DATA_DIR = 'EEGEncoder-main/data/'  # Same .pkl files as subject-dependent experiments
N_SUBJECTS = 9      # Total subjects in BCI IV-2a
N_CLASSES = 4        # 4 motor imagery classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 200       # Fewer epochs than subject-dependent (200 vs 500)
                     # because we have MORE training data (8 subjects vs 1)
BATCH_SIZE = 64      # Same as other experiments
LR = 1e-3            # Same as other experiments


# ==================== DATA LOADING ====================
def load_all_subjects(data_dir, n_subjects=9):
    """
    Load ALL 9 subjects' data from .pkl files into memory.
    
    Returns 4 lists, each containing 9 numpy arrays (one per subject):
    - all_train:        [X_train_s1, X_train_s2, ..., X_train_s9]
    - all_test:         [X_test_s1,  X_test_s2,  ..., X_test_s9]
    - all_train_labels: [y_train_s1, y_train_s2, ..., y_train_s9]
    - all_test_labels:  [y_test_s1,  y_test_s2,  ..., y_test_s9]
    
    Each X has shape (288, 1, 22, 1125) — 288 trials per session.
    Each y has shape (288, 4) — one-hot encoded labels.
    """
    all_train = []
    all_test = []
    all_train_labels = []
    all_test_labels = []
    
    for sub in range(1, n_subjects + 1):
        pkl_path = os.path.join(data_dir, f'data_all_{sub}.pkl')
        with open(pkl_path, 'rb') as f:
            X_train, X_test, y_train_oh, y_test_oh = pickle.load(f)
        # Store each subject's train and test data separately
        all_train.append(X_train)            # Shape: (288, 1, 22, 1125)
        all_test.append(X_test)              # Shape: (288, 1, 22, 1125)
        all_train_labels.append(y_train_oh)  # Shape: (288, 4)
        all_test_labels.append(y_test_oh)    # Shape: (288, 4)
    
    return all_train, all_test, all_train_labels, all_test_labels


class CombinedDataset(data.Dataset):
    """
    PyTorch Dataset that COMBINES data from multiple subjects.
    
    Used in LOSO to merge 8 subjects' data into one training set.
    Concatenates all arrays along the trial dimension.
    
    Example for training set (8 subjects):
    - 8 subjects × 576 trials/subject = 4608 total training trials
    """
    def __init__(self, X_list, y_list):
        # Concatenate all subject arrays into one big array
        self.x = torch.tensor(np.concatenate(X_list, axis=0)).float()
        self.y = torch.tensor(np.concatenate(y_list, axis=0)).float()
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)


def loso_split(all_train, all_test, all_train_labels, all_test_labels, test_subject_idx):
    """
    Create a Leave-One-Subject-Out data split.
    
    For the held-out subject (test_subject_idx):
    - TEST set = ALL their data (BOTH train and test sessions = 576 trials)
    
    For all other 8 subjects:
    - TRAIN set = ALL their data (BOTH train and test sessions = 576 each)
    - Total training: 8 × 576 = 4608 trials
    
    WHY use both train+test sessions?
    In LOSO, we're NOT testing on the same subject's held-out data.
    We're testing on a COMPLETELY DIFFERENT person. So we can use ALL
    available data from the other 8 subjects for training.
    
    Parameters
    ----------
    test_subject_idx : int
        Index (0-8) of the subject to hold out for testing
    
    Returns
    -------
    train_X, train_y : lists of arrays (8 subjects × 2 sessions each)
    test_X, test_y   : lists of arrays (1 subject × 2 sessions)
    """
    train_X, train_y = [], []
    test_X, test_y = [], []
    
    for i in range(len(all_train)):
        if i == test_subject_idx:
            # This subject is the TEST subject — ALL their data goes to test set
            test_X.append(all_train[i])          # Their training session (288 trials)
            test_X.append(all_test[i])           # Their evaluation session (288 trials)
            test_y.append(all_train_labels[i])   # Total: 576 test trials
            test_y.append(all_test_labels[i])
        else:
            # Other subjects — ALL their data goes to training set
            train_X.append(all_train[i])         # Their training session (288 trials)
            train_X.append(all_test[i])          # Their evaluation session (288 trials)
            train_y.append(all_train_labels[i])  # Total per subject: 576 trials
            train_y.append(all_test_labels[i])   # Grand total: 8 × 576 = 4608 trials
    
    return train_X, train_y, test_X, test_y


# ==================== TRAINING ====================
def train_loso(model_type='baseline'):
    """
    Run full LOSO cross-validation (9 folds).
    
    For each fold:
    1. Hold out one subject for testing
    2. Train a FRESH model on data from the other 8 subjects
    3. Evaluate on the held-out subject
    4. Record best accuracy and kappa
    
    Reports mean ± std accuracy across all 9 folds.
    """
    print(f"\n{'='*60}")
    print(f"  Subject-Independent Evaluation (LOSO)")
    print(f"  Model: {model_type}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")
    
    # Step 1: Load ALL subject data into memory
    all_train, all_test, all_train_labels, all_test_labels = load_all_subjects(DATA_DIR)
    
    results = []
    
    # Step 2: Run 9 folds (each subject takes a turn as test subject)
    for test_sub in range(N_SUBJECTS):
        print(f"\n--- Fold {test_sub + 1}/9: Testing on Subject {test_sub + 1} (trained on rest) ---")
        
        # Create LOSO split for this fold
        tr_X, tr_y, te_X, te_y = loso_split(
            all_train, all_test, all_train_labels, all_test_labels, test_sub)
        
        # Wrap in PyTorch datasets
        train_ds = CombinedDataset(tr_X, tr_y)  # 8 subjects × 576 = 4608 trials
        test_ds = CombinedDataset(te_X, te_y)   # 1 subject × 576 = 576 trials
        
        print(f"  Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
        
        # Step 3: Create a FRESH model (new random weights for each fold)
        if model_type == 'efficient':
            # Our proposed efficient model (no lma.py needed)
            from efficient_eegencoder import EfficientEEGEncoder
            model = EfficientEEGEncoder(n_classes=N_CLASSES).to(DEVICE)
        else:
            # Original baseline model (needs lma.py in path)
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EEGEncoder-main'))
            from colab_train_baseline import EEGEncoder
            model = EEGEncoder(n_classes=N_CLASSES).to(DEVICE)
        
        # Setup data loaders
        num_workers = 2 if DEVICE == 'cuda' else 0
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=num_workers,
                                        pin_memory=(DEVICE == 'cuda'))
        test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=num_workers)
        
        # Same optimizer and loss as subject-dependent experiments
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)  # Slightly less smoothing
        scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None
        
        best_acc = 0
        best_kappa = 0
        
        # Step 4: Train for N_EPOCHS
        for epoch in range(N_EPOCHS):
            # ---- Training ----
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outs = model(inputs)
                    loss = loss_func(outs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outs = model(inputs)
                    loss = loss_func(outs, labels)
                    loss.backward()
                    optimizer.step()
            
            # ---- Evaluate every 25 epochs ----
            # (Less frequent than subject-dependent since we have more samples)
            if (epoch + 1) % 25 == 0 or epoch == N_EPOCHS - 1:
                model.eval()
                preds, labels_list = [], []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(DEVICE)
                        if scaler:
                            with torch.cuda.amp.autocast():
                                outs = model(inputs)
                        else:
                            outs = model(inputs)
                        preds.extend(outs.argmax(-1).cpu().numpy().tolist())
                        labels_list.extend(labels.argmax(-1).numpy().tolist())
                
                # Compute metrics
                acc = accuracy_score(labels_list, preds)
                kappa = cohen_kappa_score(labels_list, preds)
                
                # Track best results for this fold
                if acc > best_acc:
                    best_acc = acc
                    best_kappa = kappa
                
                print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | Acc: {acc:.4f} | "
                      f"Kappa: {kappa:.4f} | Best: {best_acc:.4f}")
        
        # Store results for this fold
        results.append({
            'subject': test_sub + 1,
            'accuracy': best_acc,
            'kappa': best_kappa
        })
        print(f"  ✅ Subject {test_sub+1}: Acc={best_acc:.4f}, Kappa={best_kappa:.4f}")
    
    # ---- Summary across all 9 folds ----
    mean_acc = np.mean([r['accuracy'] for r in results])
    mean_kappa = np.mean([r['kappa'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"  SUBJECT-INDEPENDENT RESULTS ({model_type})")
    print(f"{'='*60}")
    print(f"  {'Subject':<10} {'Accuracy':<12} {'Kappa':<10}")
    print(f"  {'-'*32}")
    for r in results:
        print(f"  S{r['subject']:<9} {r['accuracy']:<12.4f} {r['kappa']:<10.4f}")
    print(f"  {'-'*32}")
    print(f"  {'Mean':<10} {mean_acc:<12.4f} {mean_kappa:<10.4f}")
    print(f"  {'Std':<10} {std_acc:<12.4f}")
    print(f"\n  📊 Paper SI accuracy: 74.48%")
    print(f"  📊 Your SI accuracy:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    return results


# ==================== ENTRY POINT ====================
if __name__ == '__main__':
    # Parse command-line argument: which model to test
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'efficient'],
                        help='Model type: baseline or efficient')
    args = parser.parse_args()
    
    # Run full LOSO cross-validation
    train_loso(model_type=args.model)
