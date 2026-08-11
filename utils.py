"""utils.py — Utility functions for Generalizable ECG Classification

Organized by project phase:
    1. Data loading and label encoding
    2. Preprocessing
    3. Feature extraction
    4. Evaluation (adapted from su_utils.py, Boglind 2025)
    5. CNN-Transformer model and dataset
    6. Training and evaluation loops
"""

import numpy as np
import pandas as pd
import math
import wfdb

from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Constants

SUPERCLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# 1. Data Loading and Label Encoding

def load_raw_data(df, sampling_rate, path):
    """Load raw ECG waveforms for all records in df.

    Args:
        df: DataFrame with filename_lr / filename_hr columns
        sampling_rate: 100 or 500 Hz
        path: base path to the PTB-XL dataset
    Returns:
        np.ndarray of shape (n_records, n_samples, n_leads=12)
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def build_multilabel_vector(superclass_list, class_names=SUPERCLASSES):
    """Convert list of superclass names to binary vector.

    Args:
        superclass_list: list of str, e.g. ['MI', 'STTC']
        class_names: list of str, ordered class names
    Returns:
        np.ndarray of shape (len(class_names),), binary float32
    """
    vec = np.zeros(len(class_names), dtype=np.float32)
    for cls in superclass_list:
        if cls in class_names:
            vec[class_names.index(cls)] = 1.0
    return vec


def build_multilabel_matrix(Y_labeled, class_names=SUPERCLASSES):
    """Build binary multi-label matrix for all records.

    Args:
        Y_labeled: DataFrame with 'diagnostic_superclass' column (lists)
        class_names: list of str, ordered class names
    Returns:
        np.ndarray of shape (n_records, n_classes)
    """
    return np.vstack([
        build_multilabel_vector(classes, class_names)
        for classes in Y_labeled['diagnostic_superclass']
    ])


# 2. Preprocessing

def preprocess_ptbxl(X, sampling_rate=100):
    """Preprocess PTB-XL signals: NaN handling, filtering, normalisation.

    Pipeline:
        1. Replace NaN with 0
        2. Bandpass filter 0.5–40 Hz (order 2 Butterworth)
        3. Per-record, per-lead z-score normalisation

    Args:
        X: np.ndarray of shape (n_records, n_samples, n_leads)
        sampling_rate: int, sampling frequency in Hz
    Returns:
        X_processed: np.ndarray of same shape
    """
    X_proc = X.copy()

    # Step 1: Handle NaN values
    n_nans = np.isnan(X_proc).sum()
    if n_nans > 0:
        print(f"  Replacing {n_nans} NaN values with 0")
        X_proc = np.nan_to_num(X_proc, nan=0.0)

    # Step 2: Bandpass filter (0.5–40 Hz)
    nyq = 0.5 * sampling_rate
    low = 0.5 / nyq
    high = min(40.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype='band')

    print(f"  Applying bandpass filter: 0.5–{min(40.0, nyq * 0.99):.1f} Hz")
    for i in range(X_proc.shape[0]):
        for lead in range(X_proc.shape[2]):
            X_proc[i, :, lead] = filtfilt(b, a, X_proc[i, :, lead])

    # Step 3: Per-record, per-lead z-score normalisation
    print(f"  Applying per-record z-score normalisation")
    for i in range(X_proc.shape[0]):
        for lead in range(X_proc.shape[2]):
            mu = X_proc[i, :, lead].mean()
            sigma = X_proc[i, :, lead].std()
            if sigma > 0:
                X_proc[i, :, lead] = (X_proc[i, :, lead] - mu) / sigma
            else:
                X_proc[i, :, lead] = 0.0

    return X_proc


# 3. Feature Extraction

def extract_signal_features(X, sampling_rate=100, lead_names=LEAD_NAMES):
    """Extract time-domain and frequency-domain features per lead.

    13 features per lead × 12 leads = 156 features per record.

    Args:
        X: np.ndarray of shape (n_records, n_samples, n_leads)
        sampling_rate: int
        lead_names: list of str
    Returns:
        feat_df: pd.DataFrame of shape (n_records, n_features)
    """
    n_records, n_samples, n_leads = X.shape
    features = []

    for i in range(n_records):
        row = {}
        for lead_idx in range(n_leads):
            sig = X[i, :, lead_idx]
            lead = lead_names[lead_idx]

            # Time-domain
            row[f'{lead}_mean'] = np.mean(sig)
            row[f'{lead}_std'] = np.std(sig)
            row[f'{lead}_skew'] = skew(sig)
            row[f'{lead}_kurtosis'] = kurtosis(sig)
            row[f'{lead}_min'] = np.min(sig)
            row[f'{lead}_max'] = np.max(sig)
            row[f'{lead}_range'] = np.ptp(sig)
            row[f'{lead}_rms'] = np.sqrt(np.mean(sig ** 2))
            row[f'{lead}_zcr'] = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)

            # Frequency-domain
            freqs, psd = welch(sig, fs=sampling_rate, nperseg=min(256, n_samples))
            row[f'{lead}_dom_freq'] = freqs[np.argmax(psd)]
            freq_res = freqs[1] - freqs[0]

            for band_name, f_lo, f_hi in [('low', 0.5, 4), ('mid', 4, 15), ('high', 15, 40)]:
                mask = (freqs >= f_lo) & (freqs <= f_hi)
                row[f'{lead}_power_{band_name}'] = np.sum(psd[mask]) * freq_res

        features.append(row)

    return pd.DataFrame(features)


# 4. Evaluation

def eval_multilabel(Y_true, Y_pred, label_names):
    """Evaluate multi-label predictions.

    Args:
        Y_true: np.ndarray of shape (n, k), binary ground truth
        Y_pred: np.ndarray of shape (n, k), binary predictions
        label_names: list of str, class names
    Returns:
        metrics: dict with subset_accuracy, micro_f1, macro_f1, hamming_loss
        per_label_df: DataFrame with per-class F1 scores
    """
    metrics = {
        'subset_accuracy': accuracy_score(Y_true, Y_pred),
        'micro_f1': f1_score(Y_true, Y_pred, average='micro', zero_division=0),
        'macro_f1': f1_score(Y_true, Y_pred, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(Y_true, Y_pred),
    }
    per_class_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_label_df = pd.DataFrame({
        'label': label_names,
        'f1': per_class_f1
    }).sort_values('f1', ascending=False).reset_index(drop=True)
    return metrics, per_label_df


# 5. CNN-Transformer Model

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (transformer lecture)."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ECGCNNTransformer(nn.Module):
    """CNN-Transformer for multi-label ECG classification.

    Architecture:
        Input (batch, 12, 1000)
        → CNN encoder: 3 × [Conv1d → BatchNorm → ReLU → MaxPool]
        → Positional encoding
        → Transformer encoder: n_layers, n_heads attention heads
        → Global average pooling
        → Dropout → Linear → n_classes outputs
    """
    def __init__(self, n_leads=12, n_classes=5, d_model=256, n_heads=4,
                 n_transformer_layers=1, dim_ff=512, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # CNN Encoder
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.cnn_block3 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(2))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, n_classes))

        self._attention_weights = None

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, n_leads, n_samples) — e.g. (batch, 12, 1000)
            return_attention: if True, also return attention weights
        Returns:
            logits: (batch, n_classes)
            attn_weights: list of (batch, n_heads, seq, seq) if return_attention
        """
        # CNN: (batch, 12, 1000) → (batch, d_model, 125)
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)

        # Transpose for transformer: (batch, 125, d_model)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)

        if return_attention:
            attn_weights = []
            for layer in self.transformer.layers:
                x2, w = layer.self_attn(
                    x, x, x, need_weights=True, average_attn_weights=False)
                attn_weights.append(w.detach())
                x = layer.norm1(x + layer.dropout1(x2))
                x2 = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(x2))
            self._attention_weights = attn_weights
        else:
            x = self.transformer(x)

        # Global average pooling → classifier
        x = x.mean(dim=1)
        logits = self.classifier(x)

        if return_attention:
            return logits, attn_weights
        return logits


class ECGDataset(Dataset):
    """PyTorch dataset for 12-lead ECG multi-label classification.

    Transposes X from (n, samples, leads) → (n, leads, samples) for Conv1d.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 6. Training & Evaluation Loops

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1

    return running_loss / n_batches


def evaluate(model, loader, criterion, device, label_names=None):
    """Evaluate model on a dataset. Returns loss and metrics dict."""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            n_batches += 1
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / n_batches
    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_labels)

    metrics = {
        'loss': avg_loss,
        'subset_accuracy': accuracy_score(Y_true, Y_pred),
        'micro_f1': f1_score(Y_true, Y_pred, average='micro', zero_division=0),
        'macro_f1': f1_score(Y_true, Y_pred, average='macro', zero_division=0),
    }

    if label_names is not None:
        per_class = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        for i, name in enumerate(label_names):
            metrics[f'f1_{name}'] = per_class[i]

    return metrics