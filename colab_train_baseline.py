"""
=============================================================================
 EEGEncoder - Colab Training Script (Baseline Reproduction)
=============================================================================
 PURPOSE: Self-contained training script to reproduce the EEGEncoder paper
          results on Google Colab. Includes all model components inline.

 HOW TO USE ON COLAB:
   1. Upload lma.py to /content/ (contains LlamaForCausalLM transformer)
   2. Upload .pkl data files to Google Drive
   3. Set DATA_DIR to your Drive path
   4. Run this script

 ARCHITECTURE (from paper):
   Input (1, 22, 1125) → ConvBlock → 5 parallel branches:
     Each branch: Dropout → TCN (temporal) + Transformer (spatial) → Dense
   → Average 5 outputs → Softmax → 4 classes

 EXPECTED RESULTS (from paper):
   Subject-Dependent Mean Accuracy: ~86.46%
   Subject-Dependent Mean Kappa:    ~0.82
=============================================================================
"""

# ==================== IMPORTS ====================
import torch                                    # Deep learning framework
import torch.nn as nn                           # Neural network layers
import torch.nn.functional as F                  # Activation functions, dropout
from torch.utils import data                     # DataLoader, Dataset
import pickle                                    # Load .pkl data files
import random                                    # Python random seed
import numpy as np                               # Array operations
from sklearn.metrics import accuracy_score, cohen_kappa_score  # Evaluation metrics
from transformers import LlamaConfig             # Configuration for Llama transformer
import time                                      # Measure inference speed
import os                                        # File path operations
import sys                                       # System path for imports

# ==================== CONFIGURATION ====================
# Change these to match your Colab paths
DATA_DIR = '/content/drive/MyDrive/eegencoder/data/'  # Directory with data_all_1.pkl ... data_all_9.pkl
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
N_SUBJECTS = 9         # BCI IV-2a has 9 subjects
N_EPOCHS = 500         # Paper uses 500 epochs per subject
BATCH_SIZE = 64        # Mini-batch size
LEARNING_RATE = 1e-3   # Adam optimizer learning rate
QUICK_TEST = False     # Set True to run 1 subject, 50 epochs only (for debugging)

if QUICK_TEST:
    N_SUBJECTS = 1
    N_EPOCHS = 50
    print("🔧 QUICK TEST MODE: 1 subject, 50 epochs")

print(f"🖥️  Device: {DEVICE}")

# ==================== REPRODUCIBILITY ====================
# Setting all random seeds ensures same results every run
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP from using multiple threads

def setup_seed(seed=32):
    """Fix all random seeds for reproducible results."""
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed_all(seed)       # PyTorch GPU (all devices)
    np.random.seed(seed)                    # NumPy
    random.seed(seed)                       # Python stdlib
    torch.backends.cudnn.deterministic = True  # Make cuDNN deterministic
    torch.backends.cudnn.benchmark = True      # Still allow cuDNN auto-tuner

setup_seed(32)


# ==================== DATA LOADING ====================
def pkl_load(one_path):
    """Load a pickle file and return its contents."""
    with open(one_path, 'rb') as f:
        return pickle.load(f)

class EEGDB(data.Dataset):
    """
    PyTorch Dataset wrapper for preprocessed EEG data.
    
    Loads a .pkl file containing (X_train, X_test, y_train_onehot, y_test_onehot)
    and returns either the train or test split based on `states`.
    
    Each item returned is: (eeg_data, onehot_label)
    - eeg_data: shape (1, 22, 1125) — 1 channel dim, 22 EEG channels, 1125 time points
    - label: shape (4,) — one-hot encoded [0,0,1,0] etc.
    """
    def __init__(self, pkl_data, states):
        X_train, X_val, y_train_onehot, y_val_onehot = pkl_load(pkl_data)
        self.states = states
        if states == 'train':
            self.x = X_train        # Training EEG data
            self.y = y_train_onehot  # Training one-hot labels
        else:
            self.x = X_val           # Validation/test EEG data
            self.y = y_val_onehot    # Validation/test one-hot labels
        self.x = torch.tensor(self.x)  # Convert numpy to pytorch tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# ==================== MODEL COMPONENTS ====================

# --- Import the Llama-based transformer ---
# lma.py contains a modified LlamaForCausalLM used as the "Stable Transformer"
# in the paper. It must be in the Python path.
sys.path.insert(0, '.')
from lma import LlamaForCausalLM


# --- L2 Regularization Wrapper Layers ---
# These wrap standard PyTorch layers to track L2 (weight decay) loss separately.
# This allows summing L2 loss manually during training instead of using
# optimizer-level weight decay.

class LinearL2(nn.Module):
    """Linear layer with L2 regularization tracking.
    l2_loss() returns: weight_decay × sum(weights²)
    """
    def __init__(self, in_features, out_features, weight_decay=0.):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.linear(x)

    def l2_loss(self):
        """Compute L2 penalty for this layer's weights."""
        return self.weight_decay * torch.sum(self.linear.weight ** 2)


class Conv1dL2(nn.Module):
    """1D Convolution with L2 regularization tracking.
    Used in TCN blocks for temporal convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation, bias=bias, groups=groups)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.conv1(x)

    def l2_loss(self):
        return self.weight_decay * torch.sum(self.conv1.weight ** 2)


class Conv2dL2(nn.Module):
    """2D Convolution with L2 regularization tracking.
    Used in ConvBlock for spatial-temporal feature extraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation, bias=bias, groups=groups)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.conv2(x)

    def l2_loss(self):
        return self.weight_decay * torch.sum(self.conv2.weight ** 2)


# --- Downsampling Projector (ConvBlock) ---
class ConvBlock(nn.Module):
    """
    Downsampling Projector: The first stage of EEGEncoder.
    
    Transforms raw EEG input into a compact feature representation:
    Input:  (batch, 1, 22, 1125)   — 22 channels, 1125 time points
    Output: (batch, 32, 20, 1)     — 32 features, 20 time steps
    
    Architecture:
    1. Temporal conv (64×1 kernel) — captures temporal patterns across time
    2. BatchNorm → Spatial depthwise conv (1×22) — mixes EEG channels
    3. BatchNorm → ELU → AvgPool(8) — downsample time by 8x
    4. Dropout → Temporal conv (16×1) — refine temporal features
    5. BatchNorm → ELU → AvgPool(7) → Dropout — downsample time by 7x
    
    Total downsampling: 1125 → 1125/8/7 ≈ 20 time steps
    """
    def __init__(self, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22, dropout=0.3):
        super().__init__()
        F2 = F1 * D  # F2 = 32 output feature maps
        
        # Step 1: Temporal convolution — filter across time dimension
        # Kernel (64, 1) slides along time axis, extracting 64-sample patterns
        self.conv1 = Conv2dL2(1, F1, (kernLength, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Step 2: Spatial depthwise convolution — mix EEG channels
        # Kernel (1, 22) looks at all 22 channels at once for spatial filtering
        # groups=F1 means each filter operates independently (depthwise)
        self.depthwise = Conv2dL2(F1, F2, (1, in_chans), groups=F1, bias=False, weight_decay=0.009)
        self.batchnorm2 = nn.BatchNorm2d(F2)
        self.activation = nn.ELU()           # Exponential Linear Unit activation
        self.avgpool1 = nn.AvgPool2d((8, 1))  # Reduce time dimension by 8x
        self.dropout1 = nn.Dropout(dropout)
        
        # Step 3: Another temporal convolution to refine features
        self.conv2 = Conv2dL2(F2, F2, (16, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))  # Reduce time by 7x
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, 1, 22, 1125) — but we need (batch, 1, 1125, 22) for temporal conv
        x = x.permute(0, 1, 3, 2)   # → (batch, 1, 1125, 22)
        
        # Temporal filtering
        x = self.conv1(x)            # → (batch, 16, 1125, 22)
        x = self.batchnorm1(x)
        
        # Spatial filtering (mix channels)
        x = self.depthwise(x)        # → (batch, 32, 1125, 1)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)         # → (batch, 32, 140, 1)  [1125/8 ≈ 140]
        x = self.dropout1(x)
        
        # Temporal refinement
        x = self.conv2(x)            # → (batch, 32, 140, 1)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)         # → (batch, 32, 20, 1)   [140/7 = 20]
        x = self.dropout2(x)
        return x


# --- Temporal Convolutional Network (TCN) ---
class Chomp1d(nn.Module):
    """Removes trailing padding from causal convolution output.
    Causal conv adds padding to the right; this trims it so the output
    length matches the input length.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock_(nn.Module):
    """
    Temporal Convolutional Network block — extracts LOCAL temporal patterns.
    
    Uses dilated causal convolutions with exponentially increasing dilation.
    This gives the TCN a large receptive field while keeping the model small.
    
    Example with depth=2, kernel_size=4:
    - Layer 0: dilation=1, receptive field = 4 samples
    - Layer 1: dilation=2, receptive field = 4+6 = 10 samples
    - Layer 2: dilation=4, receptive field = 4+6+12 = 22 samples
    
    This means the TCN can see patterns spanning ~22 time steps
    while using only small convolution kernels.
    """
    def __init__(self, input_dimension, depth, kernel_size, filters,
                 dropout, weight_decay=0.009, max_norm=0.6, activation='relu'):
        super().__init__()
        self.depth = depth
        self.activation = getattr(F, activation)  # Get activation function by name
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        
        # Downsample residual connection if input/output dims differ
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None
        
        # Initial convolution layers (without dilation)
        self.cn1 = nn.Sequential(
            Conv1dL2(input_dimension, filters, kernel_size, weight_decay=0.009),
            nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))
        self.cn2 = nn.Sequential(
            Conv1dL2(filters, filters, kernel_size, weight_decay=0.009),
            nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))

        # Dilated convolution blocks — each has exponentially growing dilation
        for i in range(depth - 1):
            dilation_size = 2 ** (i + 1)                    # 2, 4, 8, ...
            padding = (kernel_size - 1) * dilation_size     # Causal padding amount
            block_layers = [
                # First dilated conv in this block
                Conv1dL2(filters if i > 0 else input_dimension, filters,
                         kernel_size, stride=1, padding=padding,
                         dilation=dilation_size, weight_decay=0.009),
                Chomp1d(padding),            # Remove causal padding
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Second dilated conv in this block
                Conv1dL2(filters, filters, kernel_size, stride=1, padding=padding,
                         dilation=dilation_size, weight_decay=0.009),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            self.blocks.append(nn.Sequential(*block_layers))
        self.init_weights(max_norm)

    def init_weights(self, max_norm):
        """Initialize conv weights with Kaiming uniform and clip gradients."""
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    layer.weight.data = nn.init.kaiming_uniform_(layer.weight.data)
                    nn.utils.clip_grad_norm_(layer.parameters(), max_norm)

    def forward(self, x):
        # x shape: (batch, seq_len, features) → transpose for Conv1d
        out = x.transpose(1, 2)      # → (batch, features, seq_len)
        out = self.cn1(out)           # Initial conv layer 1
        out = self.cn2(out)           # Initial conv layer 2
        
        # Save for residual connection
        res = self.downsample(out) if self.downsample is not None else out
        
        # Apply dilated blocks with residual connections
        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(out)
                out += res                    # Add residual
            else:
                out = block(out)
                out += self.blocks[i - 1](res)  # Add transformed residual
            out = self.activation(out)
        
        return out.transpose(1, 2)    # → (batch, seq_len, features)


# --- Main EEGEncoder Model ---
class EEGEncoder(nn.Module):
    """
    EEGEncoder: Dual-Stream Temporal-Spatial (DSTS) model from the paper.
    
    The key idea: use BOTH temporal (TCN) and spatial (Transformer) pathways
    in parallel, then combine them. Multiple branches act as an ensemble.
    
    Architecture:
    Input (batch, 1, 22, 1125)
      ↓
    ConvBlock (shared) → (batch, 32, 20, 1) → reshape → (batch, 20, 32)
      ↓
    5 parallel branches, each:
      ├─ Dropout (regularization / ensemble effect)
      ├─ TCN → temporal features → take last timestep → (batch, 32)
      ├─ Llama Transformer → spatial features → mean pool → (batch, 32)
      ├─ Sum temporal + spatial features → (batch, 32)
      └─ Dense → (batch, 4) class logits
      ↓
    Average 5 branch outputs → (batch, 4)
      ↓
    Softmax → class probabilities
    
    Parameters: ~181,332
    """
    def __init__(self, n_classes=4, in_chans=22, in_samples=1125, n_windows=5,
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='elu', fuse='average'):
        super().__init__()
        self.n_windows = n_windows   # Number of parallel branches (5)
        self.fuse = fuse             # How to combine branches: 'average' or 'concat'
        F2 = eegn_F1 * eegn_D       # 32 — feature dimension throughout the model

        # Shared ConvBlock — downsamples input from 1125 to ~20 timesteps
        self.conv_block = ConvBlock(F1=eegn_F1, kernLength=eegn_kernelSize,
                                     poolSize=7, D=2, in_chans=22, dropout=eegn_dropout)
        
        # 5 independent TCN blocks (one per branch) — extract temporal features
        self.tcn_blocks = nn.ModuleList([
            TCNBlock_(F2, tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout, tcn_activation)
            for _ in range(n_windows)])
        
        # 5 independent dense classifiers (one per branch) — produce class logits
        self.dense_layers = nn.ModuleList([
            LinearL2(tcn_filters, n_classes, 0.5) for _ in range(n_windows)])
        
        # Dropout applied before each branch (ensemble regularization)
        self.aa_drop = nn.Dropout(0.3)

        # --- Stable Transformer (Llama-based) for each branch ---
        # The paper uses a modified Llama architecture as the "spatial" pathway.
        # It captures GLOBAL relationships between all time steps via attention.
        config = LlamaConfig()
        config.hidden_size = F2              # 32 — matches ConvBlock output
        config.pad_token_id = 0
        config.intermediate_size = F2 * 1    # 32 — FFN intermediate size
        config.num_hidden_layers = 2         # 2 transformer layers
        config.num_attention_heads = 2       # 2 attention heads (head_dim=16)
        config.vocab_size = 21               # Unused but required by Llama
        config.max_position_embeddings = 500 # Supports sequences up to 500
        config.type_vocab_size = 20          # Unused but required
        config.dropout_ratio = 0.3           # Dropout in transformer layers
        config.weight_decay = 0.5            # Transformer weight decay
        
        # Create 5 SEPARATE transformer instances (one per branch)
        # This means each branch learns different spatial patterns
        self.trm_block = nn.ModuleList([
            LlamaForCausalLM(config=config) for _ in range(n_windows)])

        # If using concat fusion, need a final dense layer to combine
        if fuse == 'concat':
            self.final_dense = LinearL2(n_classes * n_windows, n_classes, 0.5)

    def forward(self, x):
        # Step 1: Shared conv feature extraction
        x = self.conv_block(x)                 # (batch, 32, 20, 1)
        x = x[:, :, :, 0].permute(0, 2, 1)    # (batch, 20, 32) — sequence format
        
        # Step 2: Process through 5 parallel branches
        sw_outputs = []
        for i in range(self.n_windows):
            # Apply dropout (acts as stochastic ensemble — each branch sees different data)
            window_slice = self.aa_drop(x[:, :, :])
            
            # TCN pathway: extracts LOCAL temporal patterns
            # Returns (batch, seq_len, filters), we take the LAST timestep
            tcn_output = self.tcn_blocks[i](window_slice)
            tcn_output = tcn_output[:, -1, :]    # (batch, 32)
            
            # Transformer pathway: captures GLOBAL spatial relationships
            # Uses Llama transformer with inputs_embeds (continuous features, not tokens)
            # Returns hidden states, we take the MEAN across sequence
            trm_output = self.trm_block[i](
                inputs_embeds=window_slice, output_hidden_states=True
            ).hidden_states[-1].mean(1)          # (batch, 32)
            
            # Feature fusion: sum temporal + spatial features
            tcn_output = tcn_output + F.dropout(trm_output, 0.3)
            
            # Classification: map 32-dim features to 4-class logits
            dense_output = self.dense_layers[i](tcn_output)
            sw_outputs.append(dense_output)

        # Step 3: Aggregate branch outputs
        if self.fuse == 'average':
            # Average the 5 predictions (ensemble)
            out = torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        elif self.fuse == 'concat':
            out = torch.cat(sw_outputs, dim=1)
            out = self.final_dense(out)

        # Step 4: Convert to probabilities
        out = F.softmax(out, dim=1)
        return out


# ==================== TRAINING LOOP ====================
def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_subject(db_no, data_dir, device, n_epochs, batch_size, lr):
    """
    Train and evaluate the EEGEncoder on a single subject.
    
    This implements the paper's subject-dependent evaluation:
    - Train on subject's training session (288 trials)
    - Evaluate on subject's evaluation session (288 trials)
    
    Uses:
    - Adam optimizer with learning rate 1e-3
    - CrossEntropyLoss with label smoothing 0.2
    - Manual L2 regularization (summed from all layer wrappers)
    - Mixed precision training (FP16) on GPU via GradScaler
    - Total loss = 2 × (CrossEntropy + L2)
    """
    print(f"\n{'='*60}")
    print(f"  Subject {db_no + 1}")
    print(f"{'='*60}")
    
    # Create fresh model for this subject
    model = EEGEncoder().to(device)
    if db_no == 0:
        print(f"  Model Parameters: {count_parameters(model):,}")

    # Load subject's data
    pkl_path = os.path.join(data_dir, f'data_all_{db_no + 1}.pkl')
    train_db = EEGDB(pkl_path, states='train')  # Training session (288 trials)
    val_db = EEGDB(pkl_path, states='val')      # Evaluation session (288 trials)

    # Setup optimizer and data loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_workers = 2 if device == 'cuda' else 0  # Fewer workers for stability
    train_loader = data.DataLoader(train_db, batch_size=batch_size,
                                    drop_last=False, num_workers=num_workers,
                                    shuffle=True, pin_memory=(device == 'cuda'))
    val_loader = data.DataLoader(val_db, batch_size=64, drop_last=False,
                                  num_workers=num_workers, shuffle=False,
                                  pin_memory=(device == 'cuda'))

    # Loss function: CrossEntropy with label smoothing (reduces overconfidence)
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
    
    # Mixed precision scaler (FP16 forward pass, FP32 gradients — faster on GPU)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    best_acc = 0
    best_kappa = 0

    for e in range(n_epochs):
        # ---- Training Phase ----
        model.train()
        train_preds, train_labels = [], []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().to(device)  # EEG data to GPU
            labels = labels.to(device)           # One-hot labels to GPU

            if scaler:
                # GPU path: use mixed precision (FP16) for speed
                with torch.cuda.amp.autocast():
                    outs = model(inputs)          # Forward pass in FP16
                loss = loss_func(outs, labels)    # CrossEntropy loss
                # Manually sum L2 losses from ALL wrapped layers
                l2_loss = sum(m.l2_loss() for _, m in model.named_modules() if hasattr(m, 'l2_loss'))
                # Scale total loss by 2 (paper's training recipe)
                scaler.scale(2 * (loss + l2_loss)).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU path: standard precision
                outs = model(inputs)
                loss = loss_func(outs, labels)
                l2_loss = sum(m.l2_loss() for _, m in model.named_modules() if hasattr(m, 'l2_loss'))
                (2 * (loss + l2_loss)).backward()
                optimizer.step()

            # Collect predictions for accuracy calculation
            train_preds.extend(outs.argmax(-1).cpu().detach().numpy().tolist())
            train_labels.extend(labels.argmax(-1).cpu().detach().numpy().tolist())

        train_acc = np.round(accuracy_score(train_labels, train_preds), 4)

        # ---- Validation Phase ----
        model.eval()  # Disable dropout and batch norm updates
        val_preds, val_labels = [], []
        total_time = 0
        with torch.no_grad():  # No gradient computation needed
            for inputs, labels in val_loader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                start = time.time()
                if scaler:
                    with torch.cuda.amp.autocast():
                        outs = model(inputs)
                else:
                    outs = model(inputs)
                total_time += time.time() - start  # Track inference time
                val_preds.extend(outs.argmax(-1).cpu().numpy().tolist())
                val_labels.extend(labels.argmax(-1).cpu().numpy().tolist())

        # Compute evaluation metrics
        val_acc = np.round(accuracy_score(val_labels, val_preds), 4)
        kappa = np.round(cohen_kappa_score(val_labels, val_preds), 4)
        # Cohen's Kappa: measures agreement beyond chance (0=chance, 1=perfect)

        # Track best results
        if val_acc > best_acc:
            best_acc = val_acc
            best_kappa = kappa

        # Print progress every 50 epochs
        if (e + 1) % 50 == 0 or e == 0:
            print(f"  Epoch {e+1:3d}/{n_epochs} | TrainAcc: {train_acc:.4f} | "
                  f"ValAcc: {val_acc:.4f} | Kappa: {kappa:.4f} | "
                  f"BestAcc: {best_acc:.4f} | InfTime: {total_time:.3f}s")

    print(f"\n  ✅ Subject {db_no+1} Final: Accuracy={best_acc:.4f}, Kappa={best_kappa:.4f}")
    return best_acc, best_kappa


# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    results = []
    
    # Train on each subject independently (subject-dependent evaluation)
    for sub in range(N_SUBJECTS):
        acc, kappa = train_subject(sub, DATA_DIR, DEVICE, N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
        results.append({'subject': sub + 1, 'accuracy': acc, 'kappa': kappa})

    # ---- Print Results Summary ----
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Subject':<10} {'Accuracy':<12} {'Kappa':<10}")
    print(f"  {'-'*32}")
    for r in results:
        print(f"  S{r['subject']:<9} {r['accuracy']:<12.4f} {r['kappa']:<10.4f}")
    
    mean_acc = np.mean([r['accuracy'] for r in results])
    mean_kappa = np.mean([r['kappa'] for r in results])
    print(f"  {'-'*32}")
    print(f"  {'Mean':<10} {mean_acc:<12.4f} {mean_kappa:<10.4f}")
    print(f"\n  📊 Paper reports: Accuracy=86.46%, Kappa=0.82")
    print(f"  📊 Your results:  Accuracy={mean_acc*100:.2f}%, Kappa={mean_kappa:.2f}")
