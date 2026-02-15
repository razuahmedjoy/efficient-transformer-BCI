"""
=============================================================================
 EEGEncoder - Colab Training Script (Baseline Reproduction)
=============================================================================
 PURPOSE: Self-contained training script to reproduce the EEGEncoder paper
          results on Google Colab. Includes all model components inline.

 HOW TO USE ON COLAB:
   1. First run colab_preprocess_gdf.py to create .pkl files
   2. Then run this script
   
 EXPECTED RESULTS (from paper):
   Subject-Dependent Mean Accuracy: ~86.46%
   Subject-Dependent Mean Kappa:    ~0.82
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import pickle
import random
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from transformers import LlamaConfig
import time
import os
import sys

# ==================== CONFIGURATION ====================
# Change these to match your Colab paths
DATA_DIR = '/content/drive/MyDrive/eegencoder/data/'            # Directory with data_all_1.pkl ... data_all_9.pkl
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SUBJECTS = 9
N_EPOCHS = 500        # Paper uses 500; set to 50 for quick test
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
QUICK_TEST = False    # Set True to run 1 subject, 50 epochs only

if QUICK_TEST:
    N_SUBJECTS = 1
    N_EPOCHS = 50
    print("🔧 QUICK TEST MODE: 1 subject, 50 epochs")

print(f"🖥️  Device: {DEVICE}")

# ==================== SETUP ====================
os.environ['OMP_NUM_THREADS'] = '1'

def setup_seed(seed=32):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(32)


# ==================== DATA LOADING ====================
def pkl_load(one_path):
    with open(one_path, 'rb') as f:
        return pickle.load(f)

class EEGDB(data.Dataset):
    def __init__(self, pkl_data, states):
        X_train, X_val, y_train_onehot, y_val_onehot = pkl_load(pkl_data)
        self.states = states
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


# ==================== MODEL COMPONENTS ====================
# NOTE: lma.py (LlamaForCausalLM) must be importable.
# On Colab, upload lma.py to the working directory.
sys.path.insert(0, '.')
from lma import LlamaForCausalLM


class LinearL2(nn.Module):
    def __init__(self, in_features, out_features, weight_decay=0.):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.linear(x)

    def l2_loss(self):
        return self.weight_decay * torch.sum(self.linear.weight ** 2)


class Conv1dL2(nn.Module):
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


class ConvBlock(nn.Module):
    """Downsampling Projector: Conv layers + BN + ELU + AvgPool"""
    def __init__(self, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22, dropout=0.3):
        super().__init__()
        F2 = F1 * D
        self.conv1 = Conv2dL2(1, F1, (kernLength, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dL2(F1, F2, (1, in_chans), groups=F1, bias=False, weight_decay=0.009)
        self.batchnorm2 = nn.BatchNorm2d(F2)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = Conv2dL2(F2, F2, (16, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock_(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, input_dimension, depth, kernel_size, filters,
                 dropout, weight_decay=0.009, max_norm=0.6, activation='relu'):
        super().__init__()
        self.depth = depth
        self.activation = getattr(F, activation)
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None
        self.cn1 = nn.Sequential(
            Conv1dL2(input_dimension, filters, kernel_size, weight_decay=0.009),
            nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))
        self.cn2 = nn.Sequential(
            Conv1dL2(filters, filters, kernel_size, weight_decay=0.009),
            nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))

        for i in range(depth - 1):
            dilation_size = 2 ** (i + 1)
            padding = (kernel_size - 1) * dilation_size
            block_layers = [
                Conv1dL2(filters if i > 0 else input_dimension, filters,
                         kernel_size, stride=1, padding=padding,
                         dilation=dilation_size, weight_decay=0.009),
                Chomp1d(padding), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout),
                Conv1dL2(filters, filters, kernel_size, stride=1, padding=padding,
                         dilation=dilation_size, weight_decay=0.009),
                Chomp1d(padding), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout)
            ]
            self.blocks.append(nn.Sequential(*block_layers))
        self.init_weights(max_norm)

    def init_weights(self, max_norm):
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    layer.weight.data = nn.init.kaiming_uniform_(layer.weight.data)
                    nn.utils.clip_grad_norm_(layer.parameters(), max_norm)

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.cn1(out)
        out = self.cn2(out)
        res = self.downsample(out) if self.downsample is not None else out
        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(out)
                out += res
            else:
                out = block(out)
                out += self.blocks[i - 1](res)
            out = self.activation(out)
        return out.transpose(1, 2)


class EEGEncoder(nn.Module):
    """
    EEGEncoder: Dual-Stream Temporal-Spatial (DSTS) model.
    Architecture: ConvBlock → 5 parallel (Dropout → TCN + Transformer → Dense) → Average
    """
    def __init__(self, n_classes=4, in_chans=22, in_samples=1125, n_windows=5,
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='elu', fuse='average'):
        super().__init__()
        self.n_windows = n_windows
        self.fuse = fuse
        F2 = eegn_F1 * eegn_D

        self.conv_block = ConvBlock(F1=eegn_F1, kernLength=eegn_kernelSize,
                                     poolSize=7, D=2, in_chans=22, dropout=eegn_dropout)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock_(F2, tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout, tcn_activation)
            for _ in range(n_windows)])
        self.dense_layers = nn.ModuleList([
            LinearL2(tcn_filters, n_classes, 0.5) for _ in range(n_windows)])
        self.aa_drop = nn.Dropout(0.3)

        # Stable Transformer (Llama-based) for each branch
        config = LlamaConfig()
        config.hidden_size = F2
        config.pad_token_id = 0
        config.intermediate_size = F2 * 1
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.vocab_size = 21
        config.max_position_embeddings = 500
        config.type_vocab_size = 20
        config.dropout_ratio = 0.3
        config.weight_decay = 0.5
        self.trm_block = nn.ModuleList([
            LlamaForCausalLM(config=config) for _ in range(n_windows)])

        if fuse == 'concat':
            self.final_dense = LinearL2(n_classes * n_windows, n_classes, 0.5)

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)
        sw_outputs = []
        for i in range(self.n_windows):
            window_slice = self.aa_drop(x[:, :, :])
            # TCN pathway (temporal features)
            tcn_output = self.tcn_blocks[i](window_slice)
            tcn_output = tcn_output[:, -1, :]
            # Transformer pathway (spatial/global features)
            trm_output = self.trm_block[i](
                inputs_embeds=window_slice, output_hidden_states=True
            ).hidden_states[-1].mean(1)
            # Feature fusion: sum temporal + spatial
            tcn_output = tcn_output + F.dropout(trm_output, 0.3)
            dense_output = self.dense_layers[i](tcn_output)
            sw_outputs.append(dense_output)

        if self.fuse == 'average':
            out = torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        elif self.fuse == 'concat':
            out = torch.cat(sw_outputs, dim=1)
            out = self.final_dense(out)

        out = F.softmax(out, dim=1)
        return out


# ==================== TRAINING LOOP ====================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_subject(db_no, data_dir, device, n_epochs, batch_size, lr):
    """Train and evaluate on a single subject."""
    print(f"\n{'='*60}")
    print(f"  Subject {db_no + 1}")
    print(f"{'='*60}")
    
    model = EEGEncoder().to(device)
    if db_no == 0:
        print(f"  Model Parameters: {count_parameters(model):,}")

    pkl_path = os.path.join(data_dir, f'data_all_{db_no + 1}.pkl')
    train_db = EEGDB(pkl_path, states='train')
    val_db = EEGDB(pkl_path, states='val')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Colab-friendly: fewer workers
    num_workers = 2 if device == 'cuda' else 0
    train_loader = data.DataLoader(train_db, batch_size=batch_size,
                                    drop_last=False, num_workers=num_workers,
                                    shuffle=True, pin_memory=(device == 'cuda'))
    val_loader = data.DataLoader(val_db, batch_size=64, drop_last=False,
                                  num_workers=num_workers, shuffle=False,
                                  pin_memory=(device == 'cuda'))

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    best_acc = 0
    best_kappa = 0

    for e in range(n_epochs):
        # --- Training ---
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
                l2_loss = sum(m.l2_loss() for _, m in model.named_modules() if hasattr(m, 'l2_loss'))
                scaler.scale(2 * (loss + l2_loss)).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outs = model(inputs)
                loss = loss_func(outs, labels)
                l2_loss = sum(m.l2_loss() for _, m in model.named_modules() if hasattr(m, 'l2_loss'))
                (2 * (loss + l2_loss)).backward()
                optimizer.step()

            train_preds.extend(outs.argmax(-1).cpu().detach().numpy().tolist())
            train_labels.extend(labels.argmax(-1).cpu().detach().numpy().tolist())

        train_acc = np.round(accuracy_score(train_labels, train_preds), 4)

        # --- Validation ---
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

        # Print every 50 epochs
        if (e + 1) % 50 == 0 or e == 0:
            print(f"  Epoch {e+1:3d}/{n_epochs} | TrainAcc: {train_acc:.4f} | "
                  f"ValAcc: {val_acc:.4f} | Kappa: {kappa:.4f} | "
                  f"BestAcc: {best_acc:.4f} | InfTime: {total_time:.3f}s")

    print(f"\n  ✅ Subject {db_no+1} Final: Accuracy={best_acc:.4f}, Kappa={best_kappa:.4f}")
    return best_acc, best_kappa


# ==================== MAIN ====================
if __name__ == '__main__':
    results = []
    
    for sub in range(N_SUBJECTS):
        acc, kappa = train_subject(sub, DATA_DIR, DEVICE, N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
        results.append({'subject': sub + 1, 'accuracy': acc, 'kappa': kappa})

    # --- Results Summary ---
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
