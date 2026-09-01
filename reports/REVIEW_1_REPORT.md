# Review 1 Project Report: Classical Computer Vision & ML for Open-Set Intruder Detection

**Course / Project**: Computer Vision Case Study — Intruder Detection via Face Recognition  
**Milestone**: Review 1 — Handcrafted Feature Descriptors & Classical Machine Learning  
**Scope**: 5 Independent Feature + Classifier Pipelines with Open-Set Intruder Rejection  

---

## 1. Executive Summary & Architecture Overview

The primary objective of Review 1 is to establish a rigorous, modular, and reproducible face recognition and intruder detection system grounded strictly in **classical computer vision and classical machine learning** principles (excluding neural networks).

The system handles the **Open-Set Recognition Problem**: given a test face image, the system must not only identify known enrolled identities ($C_1, C_2, ..., C_K$) with high precision, but also reject any unseen imposter as an **"INTRUDER / UNKNOWN"**.

```
                    INPUT IMAGE / LIVE CAMERA FRAME
                                   │
                                   ▼
                       [ 1. Face Detection ]
                    (OpenCV Frontal Haar Cascade)
                                   │
                                   ▼
                       [ 2. Face Alignment ]
                     (Inter-Ocular Angle Rotation)
                                   │
                                   ▼
                   [ 3. Photometric Normalization ]
                     (128×128 Grayscale + CLAHE)
                                   │
        ┌──────────────┬───────────┴───┬──────────────┬──────────────┐
        ▼              ▼               ▼              ▼              ▼
   [Pipeline 1]   [Pipeline 2]    [Pipeline 3]   [Pipeline 4]   [Pipeline 5]
   BSIF Texture   LPQ Fourier      WLD Weber     Gabor Wavelet    Landmark
   Filter Bank       Phase        Excitation         Bank         Geometry
        │              │               │              │              │
        ▼              ▼               ▼              ▼              ▼
   Random Forest     k-NN         Logistic Reg     RBF-SVM      Decision Tree
        │              │               │              │              │
        └──────────────┴───────────────┼──────────────┴──────────────┘
                                       ▼
                       [ 4. Open-Set Rejection Gate ]
                       (Confidence & Distance Thresholds)
                                       │
                                       ▼
                       [ 5. Multi-Pipeline Consensus ]
                                       │
                                       ▼
                      [ AUTHORIZED vs INTRUDER ALERT ]
```

---

## 2. Independent Feature Representation & Pipeline Allocation

The framework establishes 5 completely independent feature extractors paired with tailored classical classifiers:

| Pipeline | Feature Extractor | Descriptor Family | Mathematical Basis | Classical Classifier |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline 1** | **BSIF** | Statistical Texture | Independent Component Analysis (ICA) Filter Responses | **Random Forest** (150 Trees) |
| **Pipeline 2** | **LPQ** | Fourier-Domain Phase | 2D Short-Time Fourier Transform (STFT) Phase | **k-NN** (Distance-Weighted, $k=5$) |
| **Pipeline 3** | **WLD** | Contrast / Weber Ratio | Weber's Law Differential Excitation ($\xi$) + Gradient Angle ($\theta$) | **Logistic Regression** (Multinomial L2) |
| **Pipeline 4** | **Gabor Bank** | Spatial Frequency / Orientation | 2D Gabor Wavelet Responses ($5\text{ scales} \times 8\text{ orientations}$) | **RBF-SVM** ($C=10.0, \gamma=\text{scale}$) |
| **Pipeline 5** | **Landmark Geometry** | Cranial Morphology | Scale-Invariant Distances, Aspect Ratios, Angles & Bilateral Symmetry | **Decision Tree** (CART Gini) |

---

## 3. Mathematical Formulations of Feature Descriptors

### 3.1 Pipeline 1: Binarized Statistical Image Features (BSIF)
- **Mathematical Principle**: BSIF replaces heuristic binarization rules with statistical basis filters learned from natural image patches via Independent Component Analysis (ICA).
- **Formulation**:
  Given an image patch $I(x, y)$ and $n$ orthonormal linear filters $\{W_1, W_2, \dots, W_n\}$ of dimension $l \times l$:
  $$s_i(x, y) = \sum_{u, v} W_i(u, v) I(x - u, y - v)$$
  Each response is binarized at zero:
  $$b_i(x, y) = \begin{cases} 1 & \text{if } s_i(x, y) > 0 \\ 0 & \text{otherwise} \end{cases}$$
  The $n$-bit binary string is mapped to an integer code:
  $$\text{code}(x, y) = \sum_{i=0}^{n-1} b_i(x, y) 2^i \in [0, 2^n - 1]$$
  The face is partitioned into a $4 \times 4$ grid of spatial sub-blocks. For each block, a 256-bin histogram is computed and concatenated into a global feature vector.
- **Physical Meaning**: Quantifies micro-textures (cheek skin, forehead patterns, beard/stubble, eye wrinkles).

---

### 3.2 Pipeline 2: Local Phase Quantization (LPQ)
- **Mathematical Principle**: Exploits the phase-invariance property of the 2D Short-Time Fourier Transform (STFT) under centrally symmetric blur (motion, focus blur).
- **Formulation**:
  For local window $N_x$ of size $M \times M$ around pixel $x$:
  $$F(u, x) = \sum_{y \in N_x} I(y) e^{-j 2\pi u^T y}$$
  Evaluated at 4 low frequencies: $u_1 = [a, 0]^T$, $u_2 = [0, a]^T$, $u_3 = [a, a]^T$, $u_4 = [a, -a]^T$ ($a = 1/M$).
  Separating real and imaginary components yields an 8-dimensional vector:
  $$F_x = [\text{Re}(F_1), \text{Im}(F_1), \dots, \text{Re}(F_4), \text{Im}(F_4)]^T$$
  Decorrelated via whitening transformation $G_x = V^T F_x$ and quantized:
  $$q_j(x) = \begin{cases} 1 & \text{if } G_{x, j} \ge 0 \\ 0 & \text{otherwise} \end{cases}, \quad \text{LPQ}(x) = \sum_{j=0}^7 q_j(x) 2^j$$
- **Physical Meaning**: Robust Fourier-phase signatures of structural contours around eyes, nostrils, and lips.

---

### 3.3 Pipeline 3: Weber Local Descriptor (WLD)
- **Mathematical Principle**: Grounded in Weber’s Law of human visual perception ($\Delta I / I = \text{constant}$).
- **Formulation**:
  1. **Differential Excitation ($\xi$)**:
     $$\xi(x_c) = \arctan\left[ \sum_{i=0}^{p-1} \frac{x_i - x_c}{x_c + \epsilon} \right] \in \left[ -\frac{\pi}{2}, \frac{\pi}{2} \right]$$
     - $\xi > 0$: Valley / dark spot relative to surroundings.
     - $\xi < 0$: Peak / highlight relative to surroundings.
  2. **Gradient Orientation ($\theta$)**:
     $$\theta(x_c) = \arctan2(x_7 - x_3, x_5 - x_1) \in [-\pi, \pi]$$
     Quantized into $T=8$ dominant orientation bins.
  3. **Joint 2D Histogram**: Combines differential excitation distributions across orientation sub-bands in a $4 \times 4$ spatial block grid.
- **Physical Meaning**: Illumination-invariant contrast boundaries (nose bridge, eyebrow-forehead transition, lip-skin margin).

---

### 3.4 Pipeline 4: Gabor Wavelet Filter Bank
- **Mathematical Principle**: Models the receptive field properties of mammalian primary visual cortex (V1) simple cells.
- **Formulation**:
  A 2D Gabor wavelet is defined as:
  $$\psi_{\mu, \nu}(z) = \frac{\Vert k_{\mu, \nu} \Vert^2}{\sigma^2} e^{-\frac{\Vert k_{\mu, \nu} \Vert^2 \Vert z \Vert^2}{2\sigma^2}} \left[ e^{j k_{\mu, \nu}^T z} - e^{-\frac{\sigma^2}{2}} \right]$$
  where $\nu \in \{0, 1, 2, 3, 4\}$ (5 scales) and $\mu \in \{0, 1, \dots, 7\}$ (8 orientations), resulting in 40 filters.
  Magnitude response maps $M_{\mu, \nu}(x, y) = |I * \psi_{\mu, \nu}|$ are partitioned into $4 \times 4$ blocks, from which 3 statistical moments (mean magnitude, standard deviation, energy) are pooled.
- **Physical Meaning**: Directional wrinkle lines, eyebrow arches, jaw contour slopes, and nose ridge orientations.

---

### 3.5 Pipeline 5: Facial Landmark Geometry
- **Mathematical Principle**: Direct cranial morphometry and bilateral facial symmetry indices, completely independent of pixel texture.
- **Formulation** (32 Scale-Invariant Features):
  1. **Scale-Normalized Distances**:
     - Inter-ocular distance $/ W_{\text{face}}$
     - Nose width $/ W_{\text{face}}$, Nose height $/ H_{\text{face}}$
     - Mouth width $/ W_{\text{face}}$, Mouth height $/ H_{\text{face}}$
     - Jaw width $/ W_{\text{face}}$, Eye-to-nose $/ H_{\text{face}}$, Nose-to-mouth $/ H_{\text{face}}$
  2. **Proportions & Aspect Ratios**:
     - Cranial aspect ratio ($H_{\text{face}} / W_{\text{face}}$), Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Golden Ratio deviation.
  3. **Bilateral Symmetry Indices**:
     $$\text{Sym}(P_L, P_R) = \frac{\big| |P_L.x - X_{\text{mid}}| - |P_R.x - X_{\text{mid}}| \big|}{|P_L.x - X_{\text{mid}}| + |P_R.x - X_{\text{mid}}| + \epsilon}$$
     Evaluated for eyes, eyebrows, cheeks, and mouth corners.
  4. **Angular Orientations**: Eye line tilt $\theta_{\text{eye}}$, left/right jaw angles, nose bridge slope.
- **Physical Meaning**: Bone structure, cranial shape, and geometric facial proportions.

---

## 4. Open-Set Intruder Rejection Mechanism

To prevent unauthorized strangers from being misclassified as enrolled users:

1. **Probability Gating**:
   $$\text{Decision}(x) = \begin{cases} \arg\max_c P(y=c|x) & \text{if } \max_c P(y=c|x) \ge \tau_{\text{prob}} \\ \text{"INTRUDER"} & \text{otherwise} \end{cases}$$
2. **Metric k-NN Distance Gating**:
   $$\text{Decision}(x) = \begin{cases} \text{MajorityVote} & \text{if } \frac{1}{k}\sum_{j=1}^k d(x, x_{(j)}) \le \tau_{\text{dist}} \\ \text{"INTRUDER"} & \text{otherwise} \end{cases}$$
3. **Consensus Fusion Engine**:
   All 5 pipeline predictions are polled. Access is granted as **AUTHORIZED: <Name>** only if a minimum of 2 or 3 out of 5 independent pipelines agree with confidence. Otherwise, a high-priority **INTRUDER ALERT** is generated.

---

## 5. Experimental Results & Performance Analysis

### 5.1 1-to-N Classical ML Matrix: 25 Experiments (5 Feature Descriptors $\times$ 5 Classifiers)

| Feature Extractor | Classifier | Feature Dim | Train Acc (%) | Val Acc (%) | Test Acc (%) | Precision | Recall | F1-Score | Latency (ms/face) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BSIF** | **Random Forest** | $4096\text{-D}$ | 100.0% | 59.7% | **73.6%** | 0.8286 | 0.7222 | 0.7326 | 7.03 ms |
| **BSIF** | **KNN** | $4096\text{-D}$ | 100.0% | 73.6% | **66.7%** | 0.7800 | 0.6444 | 0.6565 | 0.74 ms |
| **BSIF** | **Logistic Regression** | $4096\text{-D}$ | 100.0% | 81.9% | **77.8%** | 0.8483 | 0.7667 | 0.7671 | 0.12 ms |
| **BSIF** | **SVM** | $4096\text{-D}$ | 100.0% | 70.8% | **72.2%** | 0.7911 | 0.7111 | 0.7067 | 0.53 ms |
| **BSIF** | **Decision Tree** | $4096\text{-D}$ | 100.0% | 62.5% | **55.6%** | 0.5806 | 0.5278 | 0.5163 | 0.17 ms |
| **LPQ** | **Random Forest** | $4096\text{-D}$ | 100.0% | 72.2% | **63.9%** | 0.6404 | 0.6167 | 0.5991 | 10.84 ms |
| **LPQ** | **KNN** | $4096\text{-D}$ | 100.0% | 83.3% | **75.0%** | 0.7872 | 0.7278 | 0.7259 | 0.54 ms |
| **LPQ** | **Logistic Regression** | $4096\text{-D}$ | 100.0% | 81.9% | **81.9%** | 0.8444 | 0.8167 | 0.8044 | 0.01 ms |
| **LPQ** | **SVM** | $4096\text{-D}$ | 100.0% | 83.3% | **75.0%** | 0.8078 | 0.7444 | 0.7281 | 0.51 ms |
| **LPQ** | **Decision Tree** | $4096\text{-D}$ | 100.0% | 72.2% | **68.1%** | 0.6667 | 0.6611 | 0.6430 | 0.39 ms |
| **WLD** | **Random Forest** | $2048\text{-D}$ | 100.0% | 70.8% | **69.4%** | 0.7839 | 0.6667 | 0.6817 | 4.63 ms |
| **WLD** | **KNN** | $2048\text{-D}$ | 100.0% | 73.6% | **70.8%** | 0.7587 | 0.6833 | 0.6824 | 0.21 ms |
| **WLD** | **Logistic Regression** | $2048\text{-D}$ | 100.0% | 80.6% | **76.4%** | 0.8573 | 0.7444 | 0.7656 | 0.08 ms |
| **WLD** | **SVM** | $2048\text{-D}$ | 100.0% | 76.4% | **72.2%** | 0.7189 | 0.7000 | 0.6787 | 0.22 ms |
| **WLD** | **Decision Tree** | $2048\text{-D}$ | 100.0% | 79.2% | **70.8%** | 0.7444 | 0.6889 | 0.6826 | 0.37 ms |
| **Gabor** | **Random Forest** | $1920\text{-D}$ | 100.0% | 73.6% | **75.0%** | 0.7611 | 0.7278 | 0.7260 | 37.45 ms |
| **Gabor** | **KNN** | $1920\text{-D}$ | 100.0% | 70.8% | **75.0%** | 0.7883 | 0.7333 | 0.7210 | 0.66 ms |
| **Gabor** | **Logistic Regression** | $1920\text{-D}$ | 100.0% | 81.9% | **76.4%** | 0.7911 | 0.7444 | 0.7302 | 0.44 ms |
| **Gabor** | **SVM** | $1920\text{-D}$ | 100.0% | 76.4% | **76.4%** | 0.7796 | 0.7444 | 0.7215 | 0.19 ms |
| **Gabor** | **Decision Tree** | $1920\text{-D}$ | 100.0% | 80.6% | **75.0%** | 0.7806 | 0.7278 | 0.7175 | 0.34 ms |
| **Geometry** | **Random Forest** | $745\text{-D}$ | 100.0% | 73.6% | **69.4%** | 0.6836 | 0.6833 | 0.6549 | 2.88 ms |
| **Geometry** | **KNN** | $745\text{-D}$ | 100.0% | 76.4% | **83.3%** | 0.8744 | 0.8167 | 0.8102 | 0.48 ms |
| **Geometry** | **Logistic Regression** | $745\text{-D}$ | 100.0% | 86.1% | **81.9%** | 0.8417 | 0.8000 | 0.7921 | 0.35 ms |
| **Geometry** | **SVM** | $745\text{-D}$ | 100.0% | 80.6% | **76.4%** | 0.8144 | 0.7556 | 0.7428 | 0.10 ms |
| **Geometry** | **Decision Tree** | $745\text{-D}$ | 100.0% | 76.4% | **76.4%** | 0.8278 | 0.7444 | 0.7499 | 0.17 ms |

---

## 6. Confusion Matrix Artifacts

The system automatically generates high-resolution confusion matrix heatmaps saved in `reports/figures/`:
1. `reports/figures/confusion_matrices_test.png`: Complete 25-panel (5x5 grid) test confusion matrices for all feature-classifier combinations.
2. `reports/figures/benchmark_25_matrix.png`: 5x5 heatmap summarizing test accuracy across all 25 experiments.
3. `reports/figures/confusion_matrices_train_vs_test.png`: 10-panel comparative heatmaps evaluating Training vs Testing confusion matrices for each canonical pipeline.

---

## 7. Progression from Review 1 to Review 2

Review 1 establishes the classical computer vision baseline. For **Review 2**, the system will introduce Deep Learning models (FaceNet / ResNet embeddings + Triplet Loss) without discarding Review 1:

$$\text{Research Question: } \textit{"How do deep learned representations compare with classical handcrafted local descriptors (BSIF, LPQ, WLD, Gabor, Geometry) in open-set intruder detection under severe blur, illumination, and pose variations?"}$$

