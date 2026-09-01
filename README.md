# Intruder Detection via Face Recognition (Review 1 — Classical CV & ML)

A modular, production-ready computer vision and machine learning framework for **Open-Set Intruder Detection** using 5 independent handcrafted classical feature extractors paired with 5 distinct classical ML classifiers.

---

### 🏛️ System Architecture

```
                    INPUT IMAGE FILE / BENCHMARK IMAGE
                                  │
                                  ▼
                      [ 1. Face Detection ]
                                  │
                                  ▼
                      [ 2. Face Alignment ]
                    (Eye-Level Horizontal Rotation)
                                  │
                                  ▼
                  [ 3. Photometric Normalization ]
                      (128×128 Grayscale + CLAHE)
                                  │
         ┌──────────────┬─────────┴───┬──────────────┬──────────────┐
         ▼              ▼             ▼              ▼              ▼
    [Pipeline 1]   [Pipeline 2]  [Pipeline 3]   [Pipeline 4]   [Pipeline 5]
    BSIF Texture   LPQ Fourier    WLD Weber     Gabor Wavelet    Landmark
    Filter Bank       Phase      Excitation         Bank         Geometry
         │              │             │              │              │
         ▼              ▼             ▼              ▼              ▼
    Random Forest     k-NN       Logistic Reg     RBF-SVM      Decision Tree
         │              │             │              │              │
         └──────────────┴─────────────┼──────────────┴──────────────┘
                                      ▼
                       [ 4. Open-Set Intruder Gate ]
                       (Probabilistic / Distance Rejection)
                                      │
                                      ▼
                       [ 5. Multi-Pipeline Fusion ]
                                      │
                                      ▼
                      [ AUTHORIZED vs INTRUDER ALERT ]
```

---

## 👥 5 Independent Classical ML Pipelines

| Pipeline | Feature Extractor | Underlying Mathematical Basis | Classical Classifier | Target Facial Information |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline 1** | **BSIF** (Binarized Statistical Image Features) | Statistical filter bank learned from natural image statistics (ICA) | **Random Forest** (Tree Ensemble) | Micro skin textures, forehead/cheek patterns, beard stubble |
| **Pipeline 2** | **LPQ** (Local Phase Quantization) | Short-Time Fourier Transform (STFT) 2D local phase quantization | **k-NN** (Distance-Weighted Nearest Neighbor) | Blur-insensitive local Fourier phase transitions (eyes, lips) |
| **Pipeline 3** | **WLD** (Weber Local Descriptor) | Weber's Law Differential Excitation ($\xi$) + Gradient Orientation ($\theta$) | **Logistic Regression** (Multinomial L2) | Local contrast transitions, illumination-invariant edges |
| **Pipeline 4** | **Gabor Wavelet Bank** | Multiscale & multi-orientation 2D Gabor wavelets ($5\times8 = 40$ filters) | **RBF-SVM** (Nonlinear Kernel SVM) | Directional contours, facial wrinkles, line orientations |
| **Pipeline 5** | **Landmark Geometry** | Cranial distances, facial aspect ratios, angles, and bilateral symmetry | **Decision Tree** (CART Gini) | Structural cranial proportions and facial geometry |

---

## 📂 Project Directory Structure

```
intruder-detection/
├── requirements.txt                   # Project dependencies
├── README.md                          # Repository overview & setup instructions
├── src/
│   ├── preprocessing/
│   │   ├── detect_align.py            # Face detection, eye alignment & CLAHE
│   │   └── dataset_loader.py          # Dataset discovery & synthetic generator
│   ├── features/
│   │   ├── bsif.py                    # Pipeline 1: BSIF feature extractor
│   │   ├── lpq.py                     # Pipeline 2: LPQ feature extractor
│   │   ├── wld.py                     # Pipeline 3: WLD feature extractor
│   │   ├── gabor.py                   # Pipeline 4: Gabor wavelet bank extractor
│   │   └── geometry.py                # Pipeline 5: Landmark geometry extractor
│   ├── classifiers/
│   │   ├── pipelines.py               # Model pipelines (RF, k-NN, LR, SVM, DT)
│   │   └── train_eval.py              # Training, validation & multiclass evaluation harness
│   └── open_set/
│       └── intruder_detector.py       # Open-set confidence thresholding & fusion
├── scripts/
│   ├── train_models.py                # Model training, validation, testing & persistence
│   ├── classify_image.py              # Single-image classification CLI & test evaluation
│   └── run_review1_benchmarks.py      # End-to-end training & benchmarking script
└── reports/
    ├── REVIEW_1_REPORT.md             # Full academic review documentation
    ├── BENCHMARK_RESULTS.md           # Benchmark metrics summary
    └── figures/                       # Generated confusion matrix heatmaps
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train, Validate & Test All 5 Models
Executes 3-way dataset partitioning (Train 70%, Val 15%, Test 15%), fits all 5 models, displays confusion matrices on unseen test data, and persists trained bundles:
```bash
python scripts/train_models.py
```

### 3. Classify an Image (Direct Single-Image Inference)
Classify any input image file across all 5 classical ML pipelines and view individual predictions, confidence scores, and multi-model consensus decision:
```bash
# Classify a specific face image
python scripts/classify_image.py --image data/raw/Aashiq/Aashiq_0001.jpg

# Classify a random unseen image from the test set
python scripts/classify_image.py --random
```

### 4. Run Complete Evaluation & Export Confusion Matrix Heatmaps
Executes all 5 pipelines across Train, Validation, and Testing sets, computes open-set intruder metrics, and saves confusion matrix plots in `reports/figures/`:
```bash
python scripts/run_review1_benchmarks.py
```
