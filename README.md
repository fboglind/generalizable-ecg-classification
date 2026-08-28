# Generalizable ECG Classification

Course project for multi-label ECG diagnosis using the PTB-XL dataset. The project compares hand-crafted signal and clinical feature baselines with a CNN-Transformer trained directly on raw 12-lead ECG signals, then studies interpretability and generalization under noise and cross-dataset transfer.

## Overview

The main task is to classify 10-second, 12-lead ECG recordings into the five PTB-XL diagnostic superclasses:

- `NORM`: normal ECG
- `MI`: myocardial infarction
- `STTC`: ST/T change
- `CD`: conduction disturbance
- `HYP`: hypertrophy

The workflow includes:

- Loading PTB-XL metadata and waveform records with `wfdb`
- Aggregating SCP-ECG diagnostic codes into superclass labels
- Using the PTB-XL stratified folds: folds 1-8 for training, fold 9 for validation, fold 10 for testing
- Preprocessing signals with NaN handling, 0.5-40 Hz bandpass filtering, and per-record/per-lead z-score normalization
- Extracting time-domain and frequency-domain features from the ECG signals
- Training baseline multi-label classifiers with logistic regression and random forest
- Training a PyTorch CNN-Transformer with `BCEWithLogitsLoss`
- Inspecting attention weights and gradient-based attributions
- Testing robustness to Gaussian noise and transfer behavior on MIT-BIH LTDB segments

## Repository Structure

```text
.
├── fredrik_boglind_group_project.ipynb        # Main project notebook
├── utils.py                                   # Data, preprocessing, modeling, evaluation helpers
├── requirements.txt                          # Conda environment export
├── processed_data/                           # Cached preprocessed arrays and extracted features
└── models/                                   # Trained model weights
```

Expected external data directories:

```text
raw_data/
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
├── ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/
└── mit-bih-long-term-ecg-database-1.0.0/
```

The raw datasets are not included in this repository. Download them from PhysioNet and place them under `raw_data/` using the directory names above.

## Setup

Create the environment from the exported conda file:

```bash
conda create --name ecg-classification --file requirements.txt
conda activate ecg-classification
```

If the exported environment is too platform-specific for your machine, install the core dependencies manually:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn wfdb torch torchvision jupyter
```

## Running the Project

Start Jupyter from the project root:

```bash
jupyter lab
```

Open `fredrik_boglind_group_project.ipynb` for the project analysis.

The notebook expects the PTB-XL raw data to be available even when cached processed files are present, because metadata and raw waveforms are loaded before the cache check. It uses `processed_data/X_train_proc.npy` to decide whether the full preprocessing cache is available. If that file is missing, run the preprocessing and feature extraction cells from the notebook.

The best saved model path used by the notebook is:

```text
models/cnn_transformer_735_20260328.pt
```

## Main Results

Reported test-set performance:

| Model | Features | Subset Accuracy | Micro-F1 | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | Signal features | 0.368 | 0.635 | 0.603 |
| Random Forest | Signal features | 0.440 | 0.582 | 0.455 |
| Logistic Regression | Signal + PTB-XL+ clinical features | 0.506 | 0.743 | 0.711 |
| CNN-Transformer | Raw ECG signals | 0.505 | 0.735 | 0.692 |

CNN-Transformer per-class F1 scores:

| Class | F1 |
| --- | ---: |
| `NORM` | 0.850 |
| `MI` | 0.739 |
| `STTC` | 0.727 |
| `CD` | 0.710 |
| `HYP` | 0.434 |

The CNN-Transformer reaches performance close to the clinical-feature logistic regression baseline while using raw ECG signals rather than precomputed clinical measurements.

## Notes

- The project uses 100 Hz PTB-XL recordings to keep memory use and training time manageable.
- `processed_data/X_train_proc.npy` is not included in the current repository snapshot because of its size.
- Cross-dataset transfer to MIT-BIH LTDB is qualitative and constrained by the lead mismatch: MIT-BIH LTDB has two leads, while the model is trained on 12-lead PTB-XL ECGs.
- The notebook reports graceful degradation under moderate Gaussian noise, with sharper performance loss at very low SNR values.
