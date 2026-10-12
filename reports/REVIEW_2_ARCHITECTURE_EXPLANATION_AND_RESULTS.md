# Review 2: Deep Learning Architecture Suite & Experimental Evaluation Report

> **Project:** Intelligent Intruder Detection & Edge Face Recognition System  
> **Review Scope:** Review 2 — Deep Learning Architecture Evaluation (Mandatory & Selected Models)  
> **Evaluated Architectures:** AlexNet (2012), VGG-16 (2014), GoogLeNet / Inception-v1 (2014), ResNet-50 (2015), MobileNetV2 (2018)  
> **Execution Status:** Fully Implemented, Trained, Evaluated, and Verified  
> **Dataset Distribution:** 515 preprocessed faces across 31 unique identities (30 enrolled classes + 1 held-out unknown/intruder class) partitioned per identity (70% Train / 15% Val / 15% Test)

---

## 1. Executive Summary & Comparative Matrix

In this review phase, we implemented and empirically evaluated five deep learning architectures representing the major evolutionary milestones of convolutional networks (2012–2018). Each model was trained and evaluated under an identical biometric protocol:
- **Closed-Set Face Identification:** Multi-class classification across 30 enrolled subjects.
- **Open-Set Intruder Gating:** Extraction of 512-dimensional $\ell_2$-normalized feature embeddings followed by centroid prototype cosine-distance gating against 40 held-out intruder samples.
- **Transfer Learning Protocol:** Backbones initialized with ImageNet weights and fine-tuned for 10 epochs, ensuring stable facial landmark convergence.

### Benchmark Results Table

| Architecture | Model Paradigm | Params (M) | Model Size (FP32) | Test Accuracy | Precision | Recall | Macro F1 | Open-Set ROC-AUC | Equal Error Rate (EER) | Intruder Rejection Rate | FAR | FRR | Latency (ms/frame) | Real-Time FPS | Overall Rating |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[ResNet-50](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py)** | Residual Identity Shortcuts | **24.57M** | **93.7 MB** | **87.5%** | **0.912** | **0.875** | **0.877** | **0.924** | **15.1%** | **85.0%** | **15.0%** | **15.3%** | 15.27 ms | 65.5 FPS | **Best Overall (Gold Standard)** |
| **[MobileNetV2](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py)** | Inverted Residuals & Linear Bottlenecks | **2.90M** | **11.0 MB** | **83.3%** | **0.906** | **0.833** | **0.839** | **0.913** | **16.4%** | **82.5%** | **17.5%** | **15.3%** | 17.51 ms | 57.1 FPS | **Best Edge / CCTV Camera Model** |
| **[VGG-16](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py)** | Homogeneous $3\times3$ Cascades | **32.57M** | **124.2 MB** | **83.3%** | **0.893** | **0.833** | **0.834** | **0.890** | **17.9%** | **85.0%** | **15.0%** | **20.8%** | 34.90 ms | 28.6 FPS | **High Accuracy / Heavy Latency** |
| **[GoogLeNet](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py)** | Multi-Scale Parallel Inception | **7.17M** | **27.3 MB** | **83.3%** | **0.857** | **0.833** | **0.820** | **0.883** | **22.4%** | **77.5%** | **22.5%** | **22.2%** | 10.68 ms | 93.6 FPS | **High Throughput / Good Accuracy** |
| **[AlexNet](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py)** | Large-Kernel Shallow ConvNet | **42.34M** | **161.5 MB** | **77.8%** | **0.807** | **0.778** | **0.765** | **0.830** | **23.1%** | **77.5%** | **22.5%** | **23.6%** | **7.49 ms** | **133.5 FPS** | **Fast Baseline / Weak Discrimination** |

---

## 2. Architecture Code Mapping & Performance Justifications

### Architecture 1: AlexNet (2012) — Large-Kernel Shallow Baseline

- **Overview:** Introduced by Krizhevsky et al. (ImageNet 2012). Features 5 convolutional layers and 3 dense layers. Pioneered ReLU activations, dropout, and GPU acceleration.
- **Implementation File:** [`src/models/alexnet.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py)
- **Runner Script:** [`scripts/run_alexnet.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_alexnet.py)

#### Code Mapping: Step-by-Step
1. **Input Stem & Early Receptive Fields (`AlexNetFace.features` [L55-L60](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py#L55-L60)):**  
   Large $11\times11$ convolution with stride 2 and padding 2, followed by ReLU, Local Response Normalization (LRN), and overlapping Max-Pooling ($3\times3$, stride 2).
2. **Intermediate Convolutions (`AlexNetFace.features` [L62-L78](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py#L62-L78)):**  
   $5\times5$ conv (192 channels) and three successive $3\times3$ convolutions (384, 384, 256 channels).
3. **Adaptive Spatial Aggregation (`AlexNetFace.avgpool` [L34](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py#L34)):**  
   `nn.AdaptiveAvgPool2d((6, 6))` produces a static $256 \times 6 \times 6 = 9,216$-dimensional flattened representation.
4. **Embedding Projection Block (`AlexNetFace.fc1` & `fc2` [L35-L46](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py#L35-L46)):**  
   Linear layer ($9,216 \to 4096$) with Dropout (0.5), followed by a dense projection layer ($4096 \to 512$) to extract the biometric representation.
5. **$\ell_2$ Normalization & Classification Head (`AlexNetFace.forward` [L112-L119](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py#L112-L119)):**  
   Extracts 512-D embedding, applies `F.normalize(x, p=2, dim=1)` for cosine distance computation, and feeds into `nn.Linear(512, num_classes)`.
6. **Open-Set Intruder Gating (`scripts/run_alexnet.py` [L138-L154](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_alexnet.py#L138-L154)):**  
   Calculates class prototype centroids in 512-D space, sweeps cosine distance threshold $\tau^* = 0.109$, and evaluates against intruder test samples.

#### Empirical Results
- **Test Accuracy:** 77.8% | **Macro F1:** 0.765 | **ROC-AUC:** 0.830 | **EER:** 23.1% | **Latency:** 7.49 ms
- **Artifacts:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_alexnet.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_alexnet.png)

#### Theoretical Performance Justification (Rating: Baseline)
AlexNet delivered the lowest closed-set test accuracy (77.8%) and highest error rate (EER 23.1%) because:
1. **Oversized Early Filter Kernels:** The $11\times11$ receptive field in the first layer aggressively downsamples high-frequency facial details (pupil contours, nasal creases, eyebrow texture) into coarse spatial activations.
2. **Shallow Depth:** With only 5 convolutional layers, the network cannot construct deep hierarchical facial abstractions (such as pose-invariant 3D facial geometry).
3. **Parameter Inefficiency:** Over 90% of its 42.3M parameters reside in the fully connected layers, making the feature representation vulnerable to memorization on biometric samples.

---

### Architecture 2: VGG-16 (2014) — Deep Homogeneous $3\times3$ Stacking

- **Overview:** Introduced by Simonyan & Zisserman (Visual Geometry Group, Oxford). Proven backbone for face verification via VGGFace (Parkhi et al., 2015). Replaces large kernels with deep stacks of $3\times3$ convolutions.
- **Implementation File:** [`src/models/vgg16.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py)
- **Runner Script:** [`scripts/run_vgg16.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_vgg16.py)

#### Code Mapping: Step-by-Step
1. **Input Stem & Feature Extraction (`VGG16Face.features` [L33, L51-L52](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py#L33)):**  
   Five stages of stacked $3\times3$ convolutions with Batch Normalization: `[64, 64] -> MaxPool -> [128, 128] -> MaxPool -> [256, 256, 256] -> MaxPool -> [512, 512, 512] -> MaxPool -> [512, 512, 512] -> MaxPool`.
2. **Receptive Field Synthesis:**  
   Stacking two $3\times3$ filters synthesizes an effective $5\times5$ receptive field; three $3\times3$ filters synthesize a $7\times7$ field, but incorporates three non-linear ReLU activations instead of one.
3. **Adaptive Spatial Aggregation (`VGG16Face.avgpool` [L34](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py#L34)):**  
   `nn.AdaptiveAvgPool2d((4, 4))` yields a $512 \times 4 \times 4 = 8,192$-dimensional spatial feature map.
4. **Embedding Projection Head (`VGG16Face.classifier_block` [L35-L42](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py#L35-L42)):**  
   Two-stage projection: Linear($8,192 \to 2048$) with Dropout(0.5) $\to$ ReLU $\to$ Linear($2048 \to 512$) with Dropout(0.5) $\to$ ReLU.
5. **$\ell_2$ Normalization & Classifier (`VGG16Face.forward` [L100-L107](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py#L100-L107)):**  
   Outputs unit-sphere 512-D embeddings and classifies via `nn.Linear(512, num_classes)`.
6. **Open-Set Intruder Gating (`scripts/run_vgg16.py` [L138-L154](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_vgg16.py#L138-L154)):**  
   Evaluates intruder cosine distances at optimal threshold $\tau^* = 0.036$.

#### Empirical Results
- **Test Accuracy:** 83.3% | **Macro F1:** 0.834 | **ROC-AUC:** 0.890 | **EER:** 17.9% | **Latency:** 34.90 ms
- **Artifacts:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_vgg16.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_vgg16.png)

#### Theoretical Performance Justification (Rating: Good Accuracy, Poor Latency)
VGG-16 demonstrated strong recognition (83.3% accuracy, 85.0% intruder rejection) but slow inference (34.90 ms / 28.6 FPS) because:
1. **Fine Micro-Texture Preservation:** Homogeneous $3\times3$ convolutions preserve localized micro-edges and subtle skin textures across 16 deep layers, providing rich biometric differentiation.
2. **Computational Bottleneck:** The consecutive 512-channel convolutional stages incur massive FLOPs (over 15 GFLOPs on raw images), resulting in the highest inference latency (almost $5\times$ slower than AlexNet and $2.3\times$ slower than ResNet-50).

---

### Architecture 3: GoogLeNet / Inception-v1 (2014) — Multi-Scale Parallel Filters

- **Overview:** Introduced by Szegedy et al. (Google). Architectural foundation of **FaceNet** (Schroff et al., 2015). Processes images across multiple spatial resolutions simultaneously using parallel Inception modules and $1\times1$ dimensionality reduction.
- **Implementation File:** [`src/models/googlenet.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py)
- **Runner Script:** [`scripts/run_googlenet.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_googlenet.py)

#### Code Mapping: Step-by-Step
1. **Initial Conv Stem (`GoogLeNetFace.conv1` to `maxpool2` [L86-L92](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py#L86-L92)):**  
   $7\times7$ Conv (stride 2) $\to$ MaxPool $\to$ $1\times1$ Conv reduction $\to$ $3\times3$ Conv $\to$ MaxPool.
2. **Parallel Inception Modules (`InceptionBlock` [L21-L76](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py#L21-L76)):**  
   Executes four parallel branches per block:
   - Branch 1: $1\times1$ Conv (local detail)
   - Branch 2: $1\times1$ Conv bottleneck $\to 3\times3$ Conv (medium spatial features)
   - Branch 3: $1\times1$ Conv bottleneck $\to 5\times5$ Conv (macro facial geometry)
   - Branch 4: $3\times3$ MaxPool $\to 1\times1$ Conv projection.
3. **Channel Concatenation (`InceptionBlock.forward` [L74-L76](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py#L74-L76)):**  
   Outputs of all 4 branches are concatenated along the channel dimension: `torch.cat([out1, out2, out3, out4], 1)`.
4. **Global Average Pooling (`GoogLeNetFace.avgpool` [L115](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py#L115)):**  
   Collapses the $7\times7$ spatial feature map into a 1D channel vector, removing parameter-heavy dense layers.
5. **Linear Projection to 512-D (`GoogLeNetFace.fc` [L118](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py#L118)):**  
   Projects 1024-D pooled representation into a 512-D metric embedding with dropout regularization (0.4).
6. **Open-Set Intruder Gating (`scripts/run_googlenet.py` [L138-L154](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_googlenet.py#L138-L154)):**  
   Prototype matching in multi-scale space at optimal threshold $\tau^* = 0.211$.

#### Empirical Results
- **Test Accuracy:** 83.3% | **Macro F1:** 0.820 | **ROC-AUC:** 0.883 | **EER:** 22.4% | **Latency:** 10.68 ms
- **Artifacts:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_googlenet.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_googlenet.png)

#### Theoretical Performance Justification (Rating: Good / High Efficiency)
GoogLeNet achieved 83.3% test accuracy and high throughput (10.68 ms / 93.6 FPS) with only 7.17M parameters because:
1. **Multi-Scale Scale Invariance:** The parallel $1\times1$, $3\times3$, and $5\times5$ branches extract multi-scale biometric cues simultaneously—capturing small facial features (eyes, nose tip) alongside global skull proportions in a single layer.
2. **Parameter Economy:** $1\times1$ bottleneck convolutions compress channel depth before expensive spatial filtering, keeping the model compact (27.3 MB) and highly resistant to overfitting on biometric subsets.

---

### Architecture 4: ResNet-50 (2015) — Residual Identity Highways

- **Overview:** Introduced by He et al. (Microsoft Research). Universal gold-standard backbone for modern face recognition (ArcFace, CosFace, InsightFace). Uses identity shortcut connections ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$) to train deep 50-layer networks without vanishing gradients.
- **Implementation File:** [`src/models/resnet50.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py)
- **Runner Script:** [`scripts/run_resnet50.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_resnet50.py)

#### Code Mapping: Step-by-Step
1. **Input Stem (`ResNet50Face.conv1`, `bn1`, `maxpool` [L86-L90](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py#L86-L90)):**  
   $7\times7$ Conv (stride 2, 64 filters) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool ($3\times3$, stride 2).
2. **Bottleneck Residual Stages (`BottleneckBlock` [L20-L65](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py#L20-L65)):**  
   Four residual stages containing `[3, 4, 6, 3]` bottleneck blocks (50 total layers). Each block computes:
   $$\mathcal{F}(\mathbf{x}) = \text{Conv}_{1\times1} \to \text{BN} \to \text{ReLU} \to \text{Conv}_{3\times3} \to \text{BN} \to \text{ReLU} \to \text{Conv}_{1\times1} \to \text{BN}$$
3. **Identity Residual Addition (`BottleneckBlock.forward` [L52-L65](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py#L52-L65)):**  
   Performs shortcut sum: `out = self.relu(identity + out)`. The identity path $\mathbf{x}$ propagates gradients backward without degradation.
4. **Global Average Pooling & Projection (`ResNet50Face.avgpool` & `fc_embed` [L103-L105](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py#L103-L105)):**  
   Pools $2048 \times 4 \times 4 \to 2048 \times 1 \times 1$, then projects through `Linear(2048, 512)` $\to$ `BatchNorm1d(512)` to establish compact biometric clustering.
5. **Classification & Normalization Head (`ResNet50Face.forward` [L138-L149](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py#L138-L149)):**  
   Normalizes the 512-D embedding to the unit hypersphere and passes to `nn.Linear(512, num_classes)`.
6. **Open-Set Gating (`scripts/run_resnet50.py` [L138-L154](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_resnet50.py#L138-L154)):**  
   Fits hyperspherical prototypes and separates enrolled faces from strangers at $\tau^* = 0.235$.

#### Empirical Results
- **Test Accuracy:** **87.5%** | **Macro F1:** **0.877** | **ROC-AUC:** **0.924** | **EER:** **15.1%** | **Rejection Rate:** **85.0%** | **Latency:** 15.27 ms
- **Artifacts:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_resnet50.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_resnet50.png)

#### Theoretical Performance Justification (Rating: Best Overall / Gold Standard)
ResNet-50 outperformed all architectures across closed-set accuracy (87.5%), F1-Score (0.877), and open-set ROC-AUC (0.924) because:
1. **Unimpeded Gradient Flow:** The identity skip connections create direct linear highways for backpropagation, allowing the 50-layer network to converge smoothly without gradient decay.
2. **Landmark Preservation:** Facial landmarks (eye corners, nasal bridge, jawline) can be passed directly across layers through the identity shortcut while residual blocks learn fine invariant offsets. This forms compact intra-class clusters and wide inter-class margins on the unit hypersphere.

---

### Architecture 5: MobileNetV2 (2018) — Inverted Residuals & Edge CCTV Deployment

- **Overview:** Introduced by Sandler et al. (Google). Designed for low-power edge devices and smart CCTV systems. Utilizes depthwise separable convolutions, inverted residuals (narrow-wide-narrow), and linear bottlenecks to maximize efficiency.
- **Implementation File:** [`src/models/mobilenetv2.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py)
- **Runner Script:** [`scripts/run_mobilenetv2.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_mobilenetv2.py)

#### Code Mapping: Step-by-Step
1. **Initial Conv Stem (`MobileNetV2Face.features[0]` [L79-L84](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py#L79-L84)):**  
   Standard $3\times3$ convolution with 32 filters (stride 2) followed by BatchNorm and ReLU6.
2. **Inverted Residual Blocks (`InvertedResidual` [L21-L59](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py#L21-L59)):**  
   - Step A: $1\times1$ Pointwise expansion expands low-channel input by factor $t=6$ into a higher-dimensional space.
   - Step B: $3\times3$ Depthwise convolution (`groups=hidden_dim`) filters each channel independently, reducing computational cost by $\sim 85\%$.
   - Step C: $1\times1$ Pointwise projection reduces channels back down to the target bottleneck width.
3. **Linear Bottleneck (`InvertedResidual` [L48-L51](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py#L48-L51)):**  
   Crucially removes the non-linear activation (ReLU) after the final projection, preventing manifold collapse in low-dimensional spaces.
4. **Shortcut Residual Connections (`InvertedResidual.forward` [L55-L58](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py#L55-L58)):**  
   Applies residual shortcut connection `x + self.conv(x)` when input and output dimensions match (stride = 1).
5. **Global Pooling & 512-D Embedding (`MobileNetV2Face.classifier` [L110-L118](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py#L110-L118)):**  
   Adaptive average pooling collapses features to $1\times1$, followed by a Linear projection ($1280 \to 512$) with Dropout (0.2).
6. **Open-Set Intruder Gating (`scripts/run_mobilenetv2.py` [L138-L154](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_mobilenetv2.py#L138-L154)):**  
   Evaluates open-set gating at $\tau^* = 0.283$.

#### Empirical Results
- **Test Accuracy:** 83.3% | **Macro F1:** 0.839 | **ROC-AUC:** 0.913 | **EER:** 16.4% | **Rejection Rate:** 82.5% | **Latency:** 17.51 ms
- **Model Footprint:** **2.90M Parameters** | **11.0 MB FP32** (Smallest model in suite)
- **Artifacts:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_mobilenetv2.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_mobilenetv2.png)

#### Theoretical Performance Justification (Rating: Best Edge Architecture)
MobileNetV2 achieved 83.3% test accuracy and a high 0.913 open-set ROC-AUC while using only 2.9M parameters (11.0 MB) because:
1. **Depthwise Separability:** Decoupling spatial filtering from channel projection avoids redundant parameter representation, enabling high discriminative power with 88% fewer parameters than ResNet-50.
2. **Linear Information Preservation:** Using linear bottlenecks prevents ReLU from discarding negative facial feature coordinates in compact bottleneck representations, preserving the separability of intruder features.
3. **Deployment Suitability:** Its 11 MB footprint and 17.5 ms latency make it the ideal candidate for real-time edge CCTV deployment (e.g., Raspberry Pi 4, Jetson Nano).

---

## 3. Comparative Visualizations & Deployment Analysis

### Visual Comparisons
The benchmark suite generated the following cross-model comparison visuals in `reports/figures/`:

1. **Accuracy vs. Latency vs. Model Footprint:**  
   [`reports/figures/review2_latency_vs_accuracy.png`](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/review2_latency_vs_accuracy.png)  
   *Illustrates the Pareto frontier: MobileNetV2 and ResNet-50 achieve optimal trade-offs between parameter efficiency, latency, and biometric accuracy.*

2. **Metric Comparison Bar Chart:**  
   [`reports/figures/review2_metric_comparison_bars.png`](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/review2_metric_comparison_bars.png)  
   *Direct comparison of Closed-Set Accuracy, Open-Set ROC-AUC, and Imposter Rejection Rate across all 5 models.*

3. **Per-Architecture Confusion Matrices and ROC Curves:**
   - **AlexNet:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_alexnet.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_alexnet.png)
   - **VGG-16:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_vgg16.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_vgg16.png)
   - **GoogLeNet:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_googlenet.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_googlenet.png)
   - **ResNet-50:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_resnet50.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_resnet50.png)
   - **MobileNetV2:** [Confusion Matrix](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/confusion_matrix_mobilenetv2.png) | [ROC Curve](file:///Users/mouniksai/Downloads/intruder-detection/reports/figures/roc_mobilenetv2.png)

---

## 4. Viva / Demo Execution Guide

Each architecture has an independent runner script equipped with a `--test-only` evaluation flag. This loads saved model checkpoints from `models/` directly, executing full test evaluation, metric computation, and open-set intruder gating in seconds without retraining.

### Independent Single-Model Test Commands (Demo Mode)

To evaluate any architecture independently during the viva or demo:

```bash
# 1. Evaluate AlexNet (2012)
python3 scripts/run_alexnet.py --test-only

# 2. Evaluate VGG-16 (2014)
python3 scripts/run_vgg16.py --test-only

# 3. Evaluate GoogLeNet / Inception-v1 (2014)
python3 scripts/run_googlenet.py --test-only

# 4. Evaluate ResNet-50 (2015) - Recommended Gold Standard
python3 scripts/run_resnet50.py --test-only

# 5. Evaluate MobileNetV2 (2018) - Recommended Edge/CCTV Model
python3 scripts/run_mobilenetv2.py --test-only
```

### Full Benchmark Suite & Chart Regeneration Commands

```bash
# Run all 5 architectures in test-only mode to generate the summary matrix
python3 scripts/run_all_architectures.py --test-only

# Regenerate high-resolution comparative visualization charts
python3 scripts/generate_review2_visuals.py
```

### Retraining Commands (If Retraining Is Requested)

```bash
# Retrain an individual model (e.g., ResNet-50 for 10 epochs)
python3 scripts/run_resnet50.py --epochs 10 --batch-size 16

# Retrain all 5 models sequentially
python3 scripts/run_all_architectures.py --epochs 10 --batch-size 16
```
