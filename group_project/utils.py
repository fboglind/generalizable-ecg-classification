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

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

# Constants
SUPERCLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# 1. Data Loading & Label Encoding

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

# 5. Training
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

# 7. Interpretability

def get_attention_for_sample(model, dataset, idx, device):
    """Extract attention weights for a single sample.
    
    Returns:
        logits: (n_classes,) raw logits
        attn: (n_heads, seq_len, seq_len) attention weights
        signal: (n_leads, n_samples) raw input
    """
    model.eval()
    x, y = dataset[idx]
    x_batch = x.unsqueeze(0).to(device)  # (1, 12, 1000)
    
    with torch.no_grad():
        logits, attn_list = model(x_batch, return_attention=True)
    
    logits = logits.squeeze(0).cpu().numpy()
    attn = attn_list[0].squeeze(0).cpu().numpy()  # first (only) layer
    signal = x.cpu().numpy()
    
    return logits, attn, signal, y.numpy()

def compute_gradient_attribution(model, dataset, idx, device, class_idx):
    """Compute gradient-based attribution for a specific class.
    
    Returns:
        attribution: (n_leads, n_samples) absolute gradient magnitude
    """
    model.eval()
    x, y = dataset[idx]
    x_input = x.unsqueeze(0).to(device).requires_grad_(True)
    
    logits = model(x_input)
    logits[0, class_idx].backward()
    
    # Absolute gradient as attribution
    grad = x_input.grad.squeeze(0).abs().cpu().numpy()
    return grad

# 8. Generalizability

def add_gaussian_noise(X, snr_db):
    """Add Gaussian noise to signals at a specified SNR (in dB).
    
    SNR = 10 * log10(P_signal / P_noise)
    Higher SNR = less noise. SNR=20 is mild, SNR=5 is severe.
    """
    X_noisy = X.copy()
    for i in range(X.shape[0]):
        for lead in range(X.shape[2]):
            signal = X[i, :, lead]
            sig_power = np.mean(signal ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
            X_noisy[i, :, lead] = signal + noise
    return X_noisy

def segment_mit_bih_for_transfer(record_path, target_fs=100, window_sec=10, n_target_leads=12):
    """Load a MIT-BIH record, resample, and segment into windows.
    
    Returns:
        segments: np.ndarray (n_windows, n_samples, n_target_leads)
            — 2 real leads + 10 zero-padded leads
    """
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal  # (n_samples, 2)
    fs = record.fs
    
    # Simple resampling via decimation (128 Hz → 100 Hz approximate)
    # For proper resampling we'd use scipy.signal.resample, but for
    # a generalizability test this approximation is acceptable
    from scipy.signal import resample
    n_target_samples = int(signal.shape[0] * target_fs / fs)
    signal_resampled = resample(signal, n_target_samples, axis=0)
    
    # Segment into 10-second windows
    window_samples = target_fs * window_sec  # 1000
    n_windows = len(signal_resampled) // window_samples
    
    segments = np.zeros((n_windows, window_samples, n_target_leads), dtype=np.float32)
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        chunk = signal_resampled[start:end, :]
        
        # Place 2 MIT-BIH leads into lead positions 0 and 1 (I and II)
        # Remaining 10 leads are zero-filled
        segments[i, :, 0] = chunk[:, 0]
        segments[i, :, 1] = chunk[:, 1]
        
        # Per-lead z-score normalisation (same as preprocessing)
        for lead in range(2):
            mu = segments[i, :, lead].mean()
            sigma = segments[i, :, lead].std()
            if sigma > 0:
                segments[i, :, lead] = (segments[i, :, lead] - mu) / sigma
    
    return segments, record.sig_name
