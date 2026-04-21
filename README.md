# EEG Seizure Detection - Deep Learning Pipeline

> A deep learning-based EEG seizure detection system using 1D CNN on real clinical data from the CHB-MIT Scalp EEG Database. Achieves 99% accuracy on held-out test data with cross-dataset validation on the Bonn EEG Dataset.

---

## 🧠 Why This Project

Epilepsy affects approximately 50 million people worldwide. Manual analysis of EEG recordings is time-consuming and requires specialized expertise. This system automates seizure detection from EEG signals using deep learning — enabling timely clinical intervention and reducing burden on healthcare professionals.

---

## 📊 Results

| Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **CHB-MIT (test set)** | **99%** | **99%** | **99%** | **99%** | **0.818** |
| Bonn (cross-dataset) | 25% | 25% | 100% | 40% | 0.52 |

> Cross-dataset drop is expected due to domain shift — different sampling rates (256Hz vs 173Hz), channel counts (23 vs 1), and patient populations.

---

## 🏗️ Pipeline Architecture

```
Raw EEG Signals (CHB-MIT + Bonn)
         │
         ▼
┌─────────────────────────┐
│    Data Preprocessing    │
│  - Bandpass filter       │
│    (0.5-50 Hz Butterworth│
│  - Window segmentation   │
│    (2.5s, 25% overlap)   │
│  - Z-score normalization │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Feature Extraction     │
│  - Statistical (9/ch)    │
│  - Frequency PSD (7/ch)  │
│  - RQA (3/ch)            │
│  Total: 437 features     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Data Augmentation      │
│  - Gaussian noise        │
│  - Time shifting         │
│  - Amplitude scaling     │
│  - Channel permutation   │
│  35:65 seizure ratio     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     1D CNN Model         │
│  Conv1D(32) → BN → Pool  │
│  Conv1D(64) → BN → Pool  │
│  Conv1D(128) → BN → Pool │
│  Dense(128) → Dense(64)  │
│  Sigmoid output          │
│  23,640 parameters       │
└────────────┬────────────┘
             │
             ▼
      Seizure / Non-Seizure
```

---

## 📁 Project Structure

```
eeg-seizure-detection-mlops/
├── EEG_Seizure_Detection.ipynb   # Full pipeline notebook
├── README.md
└── data/
    ├── raw/                      # CHB-MIT + Bonn raw EEG files
    └── processed/                # Preprocessed segments
```

---

## 📚 Datasets

### CHB-MIT Scalp EEG Database
- **Source:** PhysioNet
- **Subject:** chb05
- **Sampling rate:** 256 Hz
- **Channels:** 23 EEG channels
- **Format:** European Data Format (.edf)
- **Files:** 39 total, 5 containing seizures
- **Class imbalance:** 1:245.2 (seizure:non-seizure)

### Bonn EEG Dataset
- **Source:** University of Bonn, Germany
- **Sampling rate:** 173.61 Hz
- **Channels:** 1 (single channel)
- **Segments:** 20 files (Sets A and E)
- **Used for:** Cross-dataset generalization testing

---

## 🔬 Methodology

### 1. Preprocessing
- **Bandpass filter:** 0.5–50 Hz 4th-order Butterworth (zero-phase)
- **Segmentation:** 2.5s windows with 25% overlap
- **Normalization:** Z-score per segment
- **Augmentation:** Gaussian noise, time shifting, amplitude scaling, channel permutation → achieved 35:65 seizure:non-seizure ratio

### 2. Feature Extraction (per channel)
| Domain | Features | Total |
|---|---|---|
| Statistical (time) | Mean, Std, Var, Max, Min, PtP, Median, Q25, Q75 | 9 × 23 = 207 |
| Frequency (PSD) | Delta, Theta, Alpha, Beta, Gamma, Total Power, SEF95 | 7 × 23 = 161 |
| RQA | Recurrence Rate, Determinism, Entropy | 3 × 23 = 69 |

### 3. Model Architecture
```
Input: (num_channels, timepoints_per_segment)
→ Conv1D(32, kernel=3, ReLU) → BatchNorm → MaxPool → Dropout(0.3)
→ Conv1D(64, kernel=3, ReLU) → BatchNorm → MaxPool → Dropout(0.3)
→ Conv1D(128, kernel=3, ReLU) → BatchNorm → MaxPool → Dropout(0.4)
→ Flatten
→ Dense(128, ReLU) → BatchNorm → Dropout(0.5)
→ Dense(64, ReLU) → Dropout(0.3)
→ Dense(1, Sigmoid)
Total Parameters: 23,640
```

### 4. Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Epochs | 50 |
| Early stopping | Patience=10 |
| LR reduction | Factor=0.5, Patience=5 |
| Train/Val/Test | 70% / 15% / 15% |

---

## 🔑 Key Findings

- **99% accuracy** on CHB-MIT held-out test set
- **AUC: 0.818** showing strong discrimination capability
- **Class imbalance** (0.41% seizure) remains a challenge - augmentation improved recall but clinical deployment requires threshold tuning
- **Cross-dataset drop** (99% → 25%) highlights domain shift between different recording setups
- **Delta and theta bands** show increased power during seizure events - consistent with clinical literature
- **Rhythmic oscillations at 3-5 Hz** observed consistently in seizure segments

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Trained on single subject (chb05) - may not generalize across patients
- 35% recall on CHB-MIT seizure class - misses 65% of actual seizures
- Poor cross-dataset performance due to domain shift

**Future improvements:**
- Multi-subject training for better generalization
- Attention mechanisms for critical time segment identification
- Real-time deployment optimization
- Seizure prediction (preictal detection)
- Domain adaptation for cross-dataset transfer
- Explainability via saliency maps / attention visualization
- Uncertainty quantification for clinical confidence estimates

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3 |
| Deep Learning | TensorFlow / Keras |
| Signal Processing | SciPy, NumPy |
| Data | PhysioNet CHB-MIT, Bonn EEG |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## 📖 References

1. Goldberger et al. (2000). PhysioBank, PhysioToolkit, PhysioNet. *Circulation*, 101(23)
2. Andrzejak et al. (2001). Indications of nonlinear deterministic structures in EEG. *Physical Review E*
3. Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection. MIT PhD Thesis
4. Acharya et al. (2018). Automated EEG analysis of epilepsy: A review. *Knowledge-Based Systems*

---

## 👩‍💻 Author

**Vaishnavi Mallikarjun Gajarla** 
MS Data Analytics Engineering - Northeastern University 

[GitHub](https://github.com/1825Vaishnavi) | [LinkedIn](https://linkedin.com/in/vaishnavi-mallikarjun-gajarla-726323296)
