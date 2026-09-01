# Intruder Detection via Face Recognition (Review 1 — Classical CV & ML)

A modular, production-ready computer vision and machine learning system for **Open-Set Face Recognition & Intruder Detection**. Features a **1-to-N experiment matrix** evaluating **5 handcrafted classical feature extractors** across **5 classical classifiers** (25 total experiments).

---

## 🔄 End-to-End Workflow

```
[Raw Face Image]
       │
       ▼ (src/preprocessing/detect_align.py)
[1. Preprocessing & Normalization]
  • Haar cascade / MediaPipe face localization
  • Inter-pupillary horizontal rotation alignment
  • Resize to 128×128 Grayscale + CLAHE contrast equalization
       │
       ▼ (scripts/train_models.py)
[2. 3-Way Stratified Partitioning]
  • Training Set (70%): Fits classifiers & learns feature boundaries
  • Validation Set (15%): Monitors generalization & tunes parameters
  • Testing Set (15%): Completely unseen data reserved strictly for final test evaluation
  • Intruder Pool: Unknown stranger faces for open-set security benchmarking
       │
       ▼ (src/features/)
[3. Feature Extraction (Computed Once & Cached)]
  • Extracted per dataset split and reused across all classifiers
  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
  ▼              ▼              ▼              ▼              ▼
 BSIF           LPQ            WLD           Gabor         Geometry
(4096-D)       (4096-D)       (2048-D)       (1920-D)       (745-D)
  └──────────────┴──────────────┼──────────────┴──────────────┴──────────────┘
                                │
                                ▼ (src/classifiers/)
[4. 1-to-N Classifier Training & Evaluation (25 Experiments)]
  • Each of the 5 features is evaluated with all 5 classical classifiers:
    1. Random Forest (RF)        4. RBF Kernel SVM (SVM)
    2. k-Nearest Neighbors (KNN) 5. Decision Tree / ExtraTrees (DT)
    3. Logistic Regression (LR)
                                │
                                ▼ (src/open_set/intruder_detector.py)
[5. Open-Set Intruder Gating & Consensus Fusion Engine]
  • Confidence score & metric distance thresholding against enrolled templates
  • Multi-model consensus voting: AUTHORIZED vs. INTRUDER ALERT
```

---

## 🔬 Feature Descriptors: What is Extracted?

| Feature Descriptor | Module | Dim | What is Extracted & Mathematical Basis |
| :--- | :--- | :---: | :--- |
| **BSIF** | [`src/features/bsif.py`](src/features/bsif.py) | **4096-D** | 8 natural image ICA basis filters convolved with image $\to$ binarized responses mapped to 8-bit integer codes $\to$ 16 spatial block histograms ($16 \times 256$). Captures micro-textures, pores, and facial stubble. |
| **LPQ** | [`src/features/lpq.py`](src/features/lpq.py) | **4096-D** | 2D Short-Time Fourier Transform (STFT) local phase angles at 4 low frequencies $\to$ 8-bit quantized phase codes $\to$ 16 spatial block histograms. Invariant to centrally symmetric blur. |
| **WLD** | [`src/features/wld.py`](src/features/wld.py) | **2048-D** | Weber's Law Differential Excitation ($\xi = \arctan[\sum \frac{x_i - x_c}{x_c}]$) + Gradient Orientation ($\theta$) $\to$ 2D joint histograms across 16 spatial blocks. Illumination-invariant edge contrast. |
| **Gabor** | [`src/features/gabor.py`](src/features/gabor.py) | **1920-D** | 40-filter Gabor wavelet bank ($5\text{ scales} \times 8\text{ orientations}$) $\to$ mean magnitude and standard deviation energy statistics across a $4 \times 6$ spatial grid ($40 \times 2 \times 24$). Captures directional contours and wrinkle orientations. |
| **Geometry** | [`src/features/geometry.py`](src/features/geometry.py) | **745-D** | Multi-scale directional contour geometry (3 spatial scales) + canonical anatomical anchor coordinates (eyes, nose, mouth, chin, jawline) + pairwise inter-landmark distance ratios + bilateral facial symmetry indices. |

---

## 📊 25-Experiment Benchmark Results (5 Features $\times$ 5 Classifiers)

Evaluated on 31 unique enrolled subjects across strictly separated Train (70%), Val (15%), and Test (15%) partitions:

| Feature Extractor | Classifier | Dim | Train Acc | Val Acc | Test Acc | Precision | Recall | F1-Score | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BSIF** | Random Forest | 4096-D | 100.0% | 59.7% | **73.6%** | 0.8286 | 0.7222 | 0.7326 | 7.03 ms |
| **BSIF** | KNN | 4096-D | 100.0% | 73.6% | **66.7%** | 0.7800 | 0.6444 | 0.6565 | 0.74 ms |
| **BSIF** | Logistic Regression | 4096-D | 100.0% | 81.9% | **77.8%** | 0.8483 | 0.7667 | 0.7671 | 0.12 ms |
| **BSIF** | SVM | 4096-D | 100.0% | 70.8% | **72.2%** | 0.7911 | 0.7111 | 0.7067 | 0.53 ms |
| **BSIF** | Decision Tree | 4096-D | 100.0% | 62.5% | **55.6%** | 0.5806 | 0.5278 | 0.5163 | 0.17 ms |
| **LPQ** | Random Forest | 4096-D | 100.0% | 72.2% | **63.9%** | 0.6404 | 0.6167 | 0.5991 | 10.84 ms |
| **LPQ** | KNN | 4096-D | 100.0% | 83.3% | **75.0%** | 0.7872 | 0.7278 | 0.7259 | 0.54 ms |
| **LPQ** | Logistic Regression | 4096-D | 100.0% | 81.9% | **81.9%** | 0.8444 | 0.8167 | 0.8044 | 0.01 ms |
| **LPQ** | SVM | 4096-D | 100.0% | 83.3% | **75.0%** | 0.8078 | 0.7444 | 0.7281 | 0.51 ms |
| **LPQ** | Decision Tree | 4096-D | 100.0% | 72.2% | **68.1%** | 0.6667 | 0.6611 | 0.6430 | 0.39 ms |
| **WLD** | Random Forest | 2048-D | 100.0% | 70.8% | **69.4%** | 0.7839 | 0.6667 | 0.6817 | 4.63 ms |
| **WLD** | KNN | 2048-D | 100.0% | 73.6% | **70.8%** | 0.7587 | 0.6833 | 0.6824 | 0.21 ms |
| **WLD** | Logistic Regression | 2048-D | 100.0% | 80.6% | **76.4%** | 0.8573 | 0.7444 | 0.7656 | 0.08 ms |
| **WLD** | SVM | 2048-D | 100.0% | 76.4% | **72.2%** | 0.7189 | 0.7000 | 0.6787 | 0.22 ms |
| **WLD** | Decision Tree | 2048-D | 100.0% | 79.2% | **70.8%** | 0.7444 | 0.6889 | 0.6826 | 0.37 ms |
| **Gabor** | Random Forest | 1920-D | 100.0% | 73.6% | **75.0%** | 0.7611 | 0.7278 | 0.7260 | 37.45 ms |
| **Gabor** | KNN | 1920-D | 100.0% | 70.8% | **75.0%** | 0.7883 | 0.7333 | 0.7210 | 0.66 ms |
| **Gabor** | Logistic Regression | 1920-D | 100.0% | 81.9% | **76.4%** | 0.7911 | 0.7444 | 0.7302 | 0.44 ms |
| **Gabor** | SVM | 1920-D | 100.0% | 76.4% | **76.4%** | 0.7796 | 0.7444 | 0.7215 | 0.19 ms |
| **Gabor** | Decision Tree | 1920-D | 100.0% | 80.6% | **75.0%** | 0.7806 | 0.7278 | 0.7175 | 0.34 ms |
| **Geometry** | Random Forest | 745-D | 100.0% | 73.6% | **69.4%** | 0.6836 | 0.6833 | 0.6549 | 2.88 ms |
| **Geometry** | KNN | 745-D | 100.0% | 76.4% | **83.3%** | 0.8744 | 0.8167 | 0.8102 | 0.48 ms |
| **Geometry** | Logistic Regression | 745-D | 100.0% | 86.1% | **81.9%** | 0.8417 | 0.8000 | 0.7921 | 0.35 ms |
| **Geometry** | SVM | 745-D | 100.0% | 80.6% | **76.4%** | 0.8144 | 0.7556 | 0.7428 | 0.10 ms |
| **Geometry** | Decision Tree | 745-D | 100.0% | 76.4% | **76.4%** | 0.8278 | 0.7444 | 0.7499 | 0.17 ms |

---

## 🚀 Commands & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train All 25 Experiments & Serialize Models
Executes 3-way partitioning (70/15/15), trains all 25 feature-classifier combinations, displays test confusion matrices, and persists models to `models/review1_models.joblib`:
```bash
python scripts/train_models.py
```

### 3. Run Benchmark Suite & Export Figures
Executes all 25 experiments, evaluates open-set intruder metrics, and saves the 5×5 confusion matrix grid and accuracy heatmap to `reports/figures/`:
```bash
python scripts/run_review1_benchmarks.py
```

### 4. Single-Image Inference CLI
Classify any face image with individual pipeline predictions and consensus fusion decision:
```bash
# Classify a specific face image
python scripts/classify_image.py --image data/raw/Aashiq/Aashiq_0001.jpg

# Classify a random unseen image from the test set
python scripts/classify_image.py --random
```

---

## 📂 Repository Organization

```
intruder-detection/
├── requirements.txt                   # Core Python dependencies
├── README.md                          # Project overview & instructions
├── data/raw/                          # Input subject image folders
├── models/review1_models.joblib       # Serialized models & enrolled templates
├── src/
│   ├── preprocessing/
│   │   ├── detect_align.py            # Face detection, eye alignment & CLAHE
│   │   └── dataset_loader.py          # Dataset discovery & 3-way partitioner
│   ├── features/
│   │   ├── bsif.py                    # BSIF extractor (4096-D)
│   │   ├── lpq.py                     # LPQ extractor (4096-D)
│   │   ├── wld.py                     # WLD extractor (2048-D)
│   │   ├── gabor.py                   # Gabor Wavelet Bank extractor (1920-D)
│   │   └── geometry.py                # Landmark & Contour Geometry (745-D)
│   ├── classifiers/
│   │   ├── pipelines.py               # 5 classifier factories & registries
│   │   └── train_eval.py              # 1-to-N (25 experiments) evaluation engine
│   └── open_set/
│       └── intruder_detector.py       # Template distance gating & consensus fusion
├── scripts/
│   ├── train_models.py                # 25-experiment training & serialization
│   ├── classify_image.py              # Single-image inference CLI
│   └── run_review1_benchmarks.py      # End-to-end benchmark & plotting runner
└── reports/
    ├── REVIEW_1_REPORT.md             # Full academic review documentation
    ├── BENCHMARK_RESULTS.md           # 25-experiment markdown table
    └── figures/
        ├── confusion_matrices_test.png       # 5x5 test confusion matrix grid
        ├── benchmark_25_matrix.png           # 5x5 test accuracy heatmap matrix
        └── classification_result.jpg         # Annotated visual inference output
```
