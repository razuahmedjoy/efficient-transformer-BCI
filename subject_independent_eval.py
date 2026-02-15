"""
=============================================================================
 Subject-Independent Evaluation — Leave-One-Subject-Out Cross-Validation
=============================================================================
 PURPOSE: Evaluate subject independence by training on 8 subjects and testing
          on the held-out subject. This is the professor's secondary target.

 USAGE:
   python subject_independent_eval.py --model baseline   # Test original
   python subject_independent_eval.py --model efficient   # Test proposed

 PAPER REFERENCE: Subject-Independent accuracy = 74.48%
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import pickle
import os
import sys
import time
import argparse
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== CONFIG ====================
DATA_DIR = 'EEGEncoder-main/data/'
N_SUBJECTS = 9
N_CLASSES = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 200       # Fewer epochs since more training data
BATCH_SIZE = 64
LR = 1e-3


# ==================== DATA LOADING ====================
def load_all_subjects(data_dir, n_subjects=9):
    """Load all subject data from PKL files."""
    all_train = []
    all_test = []
    all_train_labels = []
    all_test_labels = []
    
    for sub in range(1, n_subjects + 1):
        pkl_path = os.path.join(data_dir, f'data_all_{sub}.pkl')
        with open(pkl_path, 'rb') as f:
            X_train, X_test, y_train_oh, y_test_oh = pickle.load(f)
        # Pool train + test from each subject for LOSO
        all_train.append(X_train)
        all_test.append(X_test)
        all_train_labels.append(y_train_oh)
        all_test_labels.append(y_test_oh)
    
    return all_train, all_test, all_train_labels, all_test_labels


class CombinedDataset(data.Dataset):
    """Dataset that combines data from multiple subjects."""
    def __init__(self, X_list, y_list):
        self.x = torch.tensor(np.concatenate(X_list, axis=0)).float()
        self.y = torch.tensor(np.concatenate(y_list, axis=0)).float()
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)


def loso_split(all_train, all_test, all_train_labels, all_test_labels, test_subject_idx):
    """
    Leave-One-Subject-Out split.
    
    Training: all data from (N-1) subjects (both their train and test sessions)
    Testing:  all data from the held-out subject (both train and test sessions)
    """
    train_X, train_y = [], []
    test_X, test_y = [], []
    
    for i in range(len(all_train)):
        if i == test_subject_idx:
            # This subject is the test subject — use ALL their data for testing
            test_X.append(all_train[i])
            test_X.append(all_test[i])
            test_y.append(all_train_labels[i])
            test_y.append(all_test_labels[i])
        else:
            # Pool train + test data from other subjects for training
            train_X.append(all_train[i])
            train_X.append(all_test[i])
            train_y.append(all_train_labels[i])
            train_y.append(all_test_labels[i])
    
    return train_X, train_y, test_X, test_y


# ==================== TRAINING ====================
def train_loso(model_type='baseline'):
    """Run full LOSO cross-validation."""
    print(f"\n{'='*60}")
    print(f"  Subject-Independent Evaluation (LOSO)")
    print(f"  Model: {model_type}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")
    
    # Load all subject data
    all_train, all_test, all_train_labels, all_test_labels = load_all_subjects(DATA_DIR)
    
    results = []
    
    for test_sub in range(N_SUBJECTS):
        print(f"\n--- Testing on Subject {test_sub + 1} (trained on rest) ---")
        
        # LOSO split
        tr_X, tr_y, te_X, te_y = loso_split(
            all_train, all_test, all_train_labels, all_test_labels, test_sub)
        
        train_ds = CombinedDataset(tr_X, tr_y)
        test_ds = CombinedDataset(te_X, te_y)
        
        print(f"  Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
        
        # Create model
        if model_type == 'efficient':
            from efficient_eegencoder import EfficientEEGEncoder
            model = EfficientEEGEncoder(n_classes=N_CLASSES).to(DEVICE)
        else:
            # Baseline needs lma.py — make sure EEGEncoder-main/ is in path
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EEGEncoder-main'))
            from colab_train_baseline import EEGEncoder
            model = EEGEncoder(n_classes=N_CLASSES).to(DEVICE)
        
        num_workers = 2 if DEVICE == 'cuda' else 0
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=num_workers,
                                        pin_memory=(DEVICE == 'cuda'))
        test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=num_workers)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None
        
        best_acc = 0
        best_kappa = 0
        
        for epoch in range(N_EPOCHS):
            # Train
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
            
            # Evaluate every 25 epochs
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
                
                acc = accuracy_score(labels_list, preds)
                kappa = cohen_kappa_score(labels_list, preds)
                
                if acc > best_acc:
                    best_acc = acc
                    best_kappa = kappa
                
                print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | Acc: {acc:.4f} | "
                      f"Kappa: {kappa:.4f} | Best: {best_acc:.4f}")
        
        results.append({
            'subject': test_sub + 1,
            'accuracy': best_acc,
            'kappa': best_kappa
        })
        print(f"  ✅ Subject {test_sub+1}: Acc={best_acc:.4f}, Kappa={best_kappa:.4f}")
    
    # Summary
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'efficient'],
                        help='Model type: baseline or efficient')
    args = parser.parse_args()
    
    train_loso(model_type=args.model)
