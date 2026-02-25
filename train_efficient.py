"""
=============================================================================
 EfficientEEGEncoder - Training Script (Proposed Model)
=============================================================================
 PURPOSE: Train the proposed EfficientEEGEncoder (same way as baseline)
          and compare results against the baseline reproduction.

 KEY DIFFERENCE FROM BASELINE:
   - Uses EfficientEEGEncoder instead of EEGEncoder
   - Does NOT need lma.py (efficient model has its own transformer)
   - Same data loading, training loop, and evaluation as baseline

 HOW TO USE:
   1. Make sure data_all_1.pkl ... data_all_9.pkl exist in DATA_DIR
   2. Make sure efficient_eegencoder.py is in the same directory
   3. Run: python train_efficient.py

 MODIFICATIONS vs Baseline:
   1. Linear Attention (O(n) vs O(n²))
   2. 3 branches instead of 5 (40% fewer params)
   3. Depthwise Separable Convolutions
   4. Shared Transformer across branches
   5. Gradient Checkpointing (optional)
=============================================================================
"""

# ==================== IMPORTS ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data                     # DataLoader, Dataset
import pickle                                    # Load .pkl data files
import random                                    # Python random seed
import numpy as np                               # Array operations
from sklearn.metrics import accuracy_score, cohen_kappa_score  # Evaluation metrics
import time                                      # Measure inference speed
import os                                        # File path operations
import sys                                       # System path for imports

# ==================== CONFIGURATION ====================
# Same hyperparameters as baseline for fair comparison
DATA_DIR = '../EEGEncoder-main/data/'   # Directory with data_all_1.pkl ... data_all_9.pkl
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SUBJECTS = 9        # All 9 subjects
N_EPOCHS = 500        # Same as baseline (500 epochs)
BATCH_SIZE = 64       # Same as baseline
LEARNING_RATE = 1e-3  # Same as baseline
QUICK_TEST = True    # Set True for debugging (1 subject, 50 epochs)

if QUICK_TEST:
    N_SUBJECTS = 1
    N_EPOCHS = 50
    print("🔧 QUICK TEST MODE: 1 subject, 50 epochs")

print(f"🖥️  Device: {DEVICE}")

# ==================== REPRODUCIBILITY ====================
os.environ['OMP_NUM_THREADS'] = '1'

def setup_seed(seed=32):
    """Fix all random seeds — same seed as baseline for fair comparison."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(32)


# ==================== DATA LOADING ====================
# Same EEGDB class as baseline — loaded data format is identical
def pkl_load(one_path):
    """Load a pickle file containing preprocessed EEG data."""
    with open(one_path, 'rb') as f:
        return pickle.load(f)

class EEGDB(data.Dataset):
    """PyTorch Dataset for preprocessed EEG data (same as baseline)."""
    def __init__(self, pkl_data, states):
        X_train, X_val, y_train_onehot, y_val_onehot = pkl_load(pkl_data)
        if states == 'train':
            self.x = X_train
            self.y = y_train_onehot
        else:
            self.x = X_val
            self.y = y_val_onehot
        self.x = torch.tensor(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# ==================== IMPORT EFFICIENT MODEL ====================
# Import our custom EfficientEEGEncoder from efficient_eegencoder.py
# This model does NOT depend on lma.py — it has its own lightweight transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efficient_eegencoder import EfficientEEGEncoder


# ==================== TRAINING ====================
def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_subject(db_no, data_dir, device, n_epochs, batch_size, lr):
    """
    Train and evaluate EfficientEEGEncoder on a single subject.
    
    Same training procedure as baseline but with two differences:
    1. Uses EfficientEEGEncoder instead of EEGEncoder
    2. No manual L2 loss summation (efficient model uses standard layers)
    
    Everything else is identical: Adam, label smoothing 0.2, 500 epochs, etc.
    """
    print(f"\n{'='*60}")
    print(f"  Subject {db_no + 1}")
    print(f"{'='*60}")

    # Create efficient model (fewer parameters than baseline)
    model = EfficientEEGEncoder(n_classes=4).to(device)
    if db_no == 0:
        print(f"  Model Parameters: {count_parameters(model):,}")

    # Load subject data (same .pkl format as baseline)
    pkl_path = os.path.join(data_dir, f'data_all_{db_no + 1}.pkl')
    train_db = EEGDB(pkl_path, states='train')
    val_db = EEGDB(pkl_path, states='val')

    # Same optimizer as baseline
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_workers = 2 if device == 'cuda' else 0
    train_loader = data.DataLoader(train_db, batch_size=batch_size,
                                    drop_last=False, num_workers=num_workers,
                                    shuffle=True, pin_memory=(device == 'cuda'))
    val_loader = data.DataLoader(val_db, batch_size=64, drop_last=False,
                                  num_workers=num_workers, shuffle=False,
                                  pin_memory=(device == 'cuda'))

    # Same loss function as baseline
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    best_acc = 0
    best_kappa = 0

    for e in range(n_epochs):
        # ---- Training Phase ----
        model.train()
        train_preds, train_labels = [], []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    outs = model(inputs)
                loss = loss_func(outs, labels)
                # NOTE: No manual L2 loss here — efficient model uses standard layers
                scaler.scale(2 * loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outs = model(inputs)
                loss = loss_func(outs, labels)
                (2 * loss).backward()
                optimizer.step()

            train_preds.extend(outs.argmax(-1).cpu().detach().numpy().tolist())
            train_labels.extend(labels.argmax(-1).cpu().detach().numpy().tolist())

        train_acc = np.round(accuracy_score(train_labels, train_preds), 4)

        # ---- Validation Phase ----
        model.eval()
        val_preds, val_labels = [], []
        total_time = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                start = time.time()
                if scaler:
                    with torch.cuda.amp.autocast():
                        outs = model(inputs)
                else:
                    outs = model(inputs)
                total_time += time.time() - start
                val_preds.extend(outs.argmax(-1).cpu().numpy().tolist())
                val_labels.extend(labels.argmax(-1).cpu().numpy().tolist())

        val_acc = np.round(accuracy_score(val_labels, val_preds), 4)
        kappa = np.round(cohen_kappa_score(val_labels, val_preds), 4)

        if val_acc > best_acc:
            best_acc = val_acc
            best_kappa = kappa

        # Print progress every 50 epochs (same frequency as baseline)
        if (e + 1) % 50 == 0 or e == 0:
            print(f"  Epoch {e+1:3d}/{n_epochs} | TrainAcc: {train_acc:.4f} | "
                  f"ValAcc: {val_acc:.4f} | Kappa: {kappa:.4f} | "
                  f"BestAcc: {best_acc:.4f} | InfTime: {total_time:.3f}s")

    print(f"\n  ✅ Subject {db_no+1} Final: Accuracy={best_acc:.4f}, Kappa={best_kappa:.4f}")
    return best_acc, best_kappa


# ==================== MAIN ====================
if __name__ == '__main__':
    results = []

    # Train on each subject independently (same as baseline)
    for sub in range(N_SUBJECTS):
        acc, kappa = train_subject(sub, DATA_DIR, DEVICE, N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
        results.append({'subject': sub + 1, 'accuracy': acc, 'kappa': kappa})

    # ---- Results Summary ----
    print(f"\n{'='*60}")
    print(f"  EFFICIENT EEGENCODER RESULTS")
    print(f"{'='*60}")
    print(f"  {'Subject':<10} {'Accuracy':<12} {'Kappa':<10}")
    print(f"  {'-'*32}")
    for r in results:
        print(f"  S{r['subject']:<9} {r['accuracy']:<12.4f} {r['kappa']:<10.4f}")

    mean_acc = np.mean([r['accuracy'] for r in results])
    mean_kappa = np.mean([r['kappa'] for r in results])
    print(f"  {'-'*32}")
    print(f"  {'Mean':<10} {mean_acc:<12.4f} {mean_kappa:<10.4f}")

    # Compare with baseline results
    print(f"\n  📊 Baseline results:  Accuracy=84.72%, Kappa=0.80")
    print(f"  📊 Efficient results: Accuracy={mean_acc*100:.2f}%, Kappa={mean_kappa:.2f}")
