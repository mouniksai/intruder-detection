# Review 2 Deep Learning Architecture Selection & Theoretical Justification

**Course / Project**: Computer Vision Case Study — Intruder Detection via Face Recognition  
**Milestone**: Review 2 — Deep Learning Architectures & Open-Set Intruder Rejection  
**Mandatory Architectures**: AlexNet, VGGNet, GoogLeNet  
**Selected Additional Architectures**: ResNet-50, MobileNetV2  

---

## 1. Executive Summary & Architecture Suite Overview

For Case Study Review 2, the system transitions from classical handcrafted feature extractors (BSIF, LPQ, WLD, Gabor, Landmark Geometry) and classical ML classifiers to end-to-end deep convolutional neural network representations.

To build an academically rigorous and operationally realistic intruder detection benchmark, the architecture suite spans the complete evolution of deep learning paradigms:

1. **AlexNet (2012)** *(Mandatory)*: Large-kernel, shallow deep CNN baseline.
2. **VGGNet / VGG-16 (2014)** *(Mandatory)*: Deep homogeneous $3\times3$ filter stacking (foundational architecture of VGGFace).
3. **GoogLeNet / Inception-v1 (2014)** *(Mandatory)*: Multi-scale parallel Inception modules and $1\times1$ dimensionality reduction.
4. **ResNet-50 (2015)** *(Selected #1)*: Deep residual learning with identity shortcut connections (the gold standard backbone for modern face biometrics like ArcFace and InsightFace).
5. **MobileNetV2 (2018)** *(Selected #2)*: Depthwise separable convolutions, inverted residuals, and linear bottlenecks tailored for real-time edge security camera deployment.

---

## 2. Comparative Architecture Specification Matrix

| Metric / Dimension | AlexNet | VGG-16 | GoogLeNet (Inception-v1) | ResNet-50 | MobileNetV2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Year / Authors** | 2012 (Krizhevsky et al.) | 2014 (Simonyan & Zisserman) | 2014 (Szegedy et al.) | 2015 (He et al.) | 2018 (Sandler et al.) |
| **Total Layers** | 8 (5 Conv + 3 FC) | 16 (13 Conv + 3 FC) | 22 (Inception modules) | 50 (Residual blocks) | 53 (Inverted residual blocks) |
| **Parameter Count** | ~61.0 Million | ~138.4 Million | ~6.8 Million | ~25.5 Million | ~3.4 Million |
| **Model Size (FP32)** | ~233 MB | ~528 MB | ~27 MB | ~98 MB | ~14 MB |
| **Key Architectural Element** | $11\times11, 5\times5$ Conv + LRN | Stacked $3\times3$ Conv + MaxPool | Inception Module ($1\times1, 3\times3, 5\times5$) | Residual Block ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$) | Depthwise Separable + Linear Bottleneck |
| **Receptive Field Growth** | Aggressive (stride 4) | Deep, sequential $3\times3$ | Multi-scale simultaneous | Deep identity-bridged | Inverted expansion / projection |
| **Surveillance Edge Suitability** | Poor | Infeasible on edge hardware | Moderate | Good (Server/GPU) | **Optimal (Edge/CCTV/Mobile)** |
| **Biometric Domain Pedigree** | Historical baseline | Foundational (**VGGFace**) | Applied in **FaceNet** | State-of-the-Art (**ArcFace**) | On-device face authentication |

---

## 3. In-Depth Justification: Why Each Architecture Worked That Way

### 3.1 AlexNet (2012) — The Large-Kernel Shallow Baseline
* **Structural Principles**:
  * 5 convolutional layers followed by 3 dense fully connected layers.
  * Large convolutional receptive fields early: Conv1 uses $11\times11$ filters with stride 4; Conv2 uses $5\times5$ filters.
  * Over 90% of model parameters reside in the fully connected layers (FC6 alone contains $4096 \times 9216 \approx 37.7\text{M}$ weights).
* **Behavior & Justification in Face Recognition**:
  * **Loss of Facial Micro-Textures**: Rapid spatial downsampling (stride 4 with $11\times11$ receptive field in layer 1) destroys high-frequency spatial details (such as eye-corner phase transitions, eyelid folds, and fine skin textures).
  * **High Overfitting Propensity**: Dense FC layers have excessive parameter freedom, causing the network to memorize specific face training images rather than learning lighting- and pose-invariant biometric manifolds.
  * **Role in Review 2**: Demonstrates the performance and capacity limitations of early deep CNNs compared to modern architectures.

---

### 3.2 VGGNet / VGG-16 (2014) — Homogeneous Depth & Micro-Texture Extraction
* **Structural Principles**:
  * Replaces large filters with cascades of small $3\times3$ convolutions (stride 1, padding 1) separated by $2\times2$ max-pooling.
  * Stacking two $3\times3$ layers yields an effective receptive field of $5\times5$; stacking three yields $7\times7$.
  * Integrates multiple non-linear activations (ReLU) across stacked layers, providing richer non-linear feature partitioning with fewer parameters ($3 \times 3^2 C^2 = 27C^2$ vs $49C^2$ for a single $7\times7$ filter).
* **Behavior & Justification in Face Recognition**:
  * **Hierarchical Facial Geometry**: Step-by-step receptive field expansion captures high-frequency edge textures in shallow layers, local facial landmarks (eyes, nose contours, lips) in intermediate layers, and holistic face templates in deep layers.
  * **Biometric Heritage**: Formed the exact foundation of **VGGFace** (Parkhi et al., 2015), validating its ability to represent facial identities.
  * **Practical Limitations**:
    1. Extreme parameter count (~138M) and memory usage (~528 MB) produce high latency unsuitable for live security feeds.
    2. Susceptible to vanishing gradients during backpropagation without residual connections.

---

### 3.3 GoogLeNet / Inception-v1 (2014) — Multi-Scale Parallel Processing
* **Structural Principles**:
  * **Inception Modules**: Executes four parallel paths simultaneously: $1\times1$ convolution, $3\times3$ convolution, $5\times5$ convolution, and $3\times3$ max-pooling, concatenating their channel outputs.
  * **$1\times1$ Dimensionality Reduction**: Compresses channel dimensions before executing expensive spatial convolutions.
  * **Global Average Pooling (GAP)**: Replaces dense FC layers with spatial average pooling, cutting parameters to just ~6.8M.
* **Behavior & Justification in Face Recognition**:
  * **Simultaneous Multi-Scale Biometrics**: Facial identification requires analyzing both local fine details (eye corners, lip edges, nostril contours via $1\times1$ and $3\times3$ branches) and macro-structural shapes (jaw width, forehead height, eye-to-chin distance via the $5\times5$ branch) within the same layer.
  * **Generalization on Small Datasets**: Replacing millions of fully connected parameters with Global Average Pooling prevents overfitting to training subjects.
  * **Biometric Heritage**: Forms the architectural backbone family used in Google's **FaceNet**.

---

### 3.4 ResNet-50 (2015) — Chosen Architecture #1: Residual Learning
* **Structural Principles**:
  * Formulates layers as learning residual mappings $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$ using **Identity Shortcut Connections**:
    $$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$
  * Gradient Highway: Gradients flow unimpeded directly back to early layers:
    $$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
* **Behavior & Justification in Face Recognition**:
  * **Preservation of Low-Level Landmark Features**: Identity connections ensure early spatial alignments and landmark coordinates are not corrupted as feature maps pass through deep non-linear layers.
  * **Deep Discriminative Representations**: Solves the degradation problem, allowing training of 50+ layer representations that maximize inter-class separation while minimizing intra-class variance.
  * **Biometric Standard**: ResNet backbones (e.g., Modified ResNet-50) are the universal standard in modern state-of-the-art face recognition systems (ArcFace, CosFace, MagFace, InsightFace).

---

### 3.5 MobileNetV2 (2018) — Chosen Architecture #2: Real-Time Edge Surveillance
* **Structural Principles**:
  * **Depthwise Separable Convolutions**: Decouples spatial filtering (depthwise $3\times3$ conv per channel) from cross-channel feature combination (pointwise $1\times1$ conv), reducing computational complexity by ~85–90%.
  * **Inverted Residual Blocks**: Employs a *narrow $\to$ wide $\to$ narrow* structure, expanding channels internally for non-linear manifold capacity and projecting back to a compact bottleneck.
  * **Linear Bottlenecks**: Removes ReLU activations from the bottleneck output to prevent destroying information in low-dimensional subspaces.
* **Behavior & Justification in Face Recognition**:
  * **Edge Surveillance Deployment**: Real-world intruder detection systems operate on edge devices (CCTV hardware, Raspberry Pi, Jetson Nano, smart doorbells). At ~3.4M parameters and ~14 MB, MobileNetV2 enables 30+ FPS real-time surveillance.
  * **Manifold Preservation**: Facial identity manifolds naturally occupy low-dimensional subspaces. Linear bottlenecks prevent the non-linear "manifold collapse" problem, retaining subtle biometric differences between subjects while remaining lightweight.

---

## 4. Open-Set Intruder Gating Pipeline for Review 2

Standard closed-set softmax classification forces unseen intruder faces into the most similar known class. Review 2 adapts the open-set rejection mechanism established in Review 1:

```
[ Input Aligned Face (128×128 / 224×224) ]
                   │
                   ▼
[ Deep CNN Backbone (AlexNet / VGG / GoogLeNet / ResNet / MobileNet) ]
                   │
                   ▼
[ L2-Normalized Feature Embedding: e = f(x) / ||f(x)|| ]
                   │
                   ▼
[ Distance Metric Gating against Enrolled Class Prototypes c_k ]
   d* = min_k ( 1 - cos(e, c_k) )
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    d* < tau             d* >= tau
   [ AUTHORIZED ]      [ INTRUDER ALERT ]
  (Class = argmin)     (Unknown / Imposter)
```

### Evaluation Protocol:
1. **Primary Dataset**: Enrolled subject faces partitioned into Train (70%), Val (15%), Test (15%).
2. **Intruder Evaluation Pool**: Held-out unknown faces (`data/raw/unknown`) never exposed to training.
3. **Metrics**:
   * Closed-Set Test Accuracy, Precision, Recall, F1-Score.
   * Open-Set **False Accept Rate (FAR)** vs. **False Reject Rate (FRR)**.
   * **ROC Curve & AUC** across threshold sweeps ($\tau \in [0.0, 1.0]$).
   * **Inference Latency (ms)** and **Model Parameter Footprint (MB)**.
