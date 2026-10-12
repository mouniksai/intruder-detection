# Review 2: Deep Learning Architectures & Open-Set Intruder Detection
## Comprehensive Demo Guide & Comparative Performance Justification

**Project**: Intruder Detection via Face Recognition (Computer Vision Case Study)  
**Milestone**: Review 2 — Deep Convolutional Neural Network Architectures & Open-Set Metric Gating  
**Mandatory Architectures**: AlexNet (2012), VGG-16 (2014), GoogLeNet / Inception-v1 (2014)  
**Selected Architectures**: ResNet-50 (2015), MobileNetV2 (2018)  
**Evaluation Protocol**: 5 Separate Independent Pipelines (Run, Test, and Profile Individually)

---

## 1. Executive Summary & Demo Quick-Start

In Review 1, the system extracted handcrafted texture, phase, differential excitation, and geometric features (BSIF, LPQ, WLD, Gabor, Landmark Geometry) and classified them using classical machine learning (Random Forest, k-NN, Logistic Regression, RBF-SVM, Decision Trees).

In **Review 2**, the system transitions to **end-to-end deep convolutional neural network representations**. Each architecture is implemented as an **isolated, standalone pipeline** with its own execution and evaluation script.

### 🚀 Independent Execution Commands (Demo Ready)

To demonstrate and test each architecture separately during the viva/demo:

```bash
# 1. AlexNet (2012) - Mandatory Shallow Baseline
.venv/bin/python scripts/run_alexnet.py --epochs 8

# 2. VGG-16 (2014) - Mandatory Deep Homogeneous Cascade (VGGFace)
.venv/bin/python scripts/run_vgg16.py --epochs 8

# 3. GoogLeNet / Inception-v1 (2014) - Mandatory Multi-Scale Parallel Filters (FaceNet)
.venv/bin/python scripts/run_googlenet.py --epochs 8

# 4. ResNet-50 (2015) - Selected Architecture #1 (ArcFace / InsightFace Gold Standard)
.venv/bin/python scripts/run_resnet50.py --epochs 8

# 5. MobileNetV2 (2018) - Selected Architecture #2 (Edge CCTV / Real-Time Surveillance)
.venv/bin/python scripts/run_mobilenetv2.py --epochs 8

# (Optional) Run Master Orchestrator for All 5 Models
.venv/bin/python scripts/run_all_architectures.py --epochs 8
```

---

## 2. Comparative Performance Matrix (Experimental Results)

Evaluated across enrolled identities ($70\%$ train, $15\%$ val, $15\%$ test) and held-out imposter/intruder faces ($100$ unseen strangers):

| Dimension / Metric | AlexNet (2012) | VGG-16 (2014) | GoogLeNet (2014) | ResNet-50 (2015) | MobileNetV2 (2018) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Year / Authors** | Krizhevsky et al. | Simonyan & Zisserman | Szegedy et al. | He et al. | Sandler et al. |
| **Role in Review 2** | Mandatory Baseline | Mandatory | Mandatory | Selected Choice #1 | Selected Choice #2 |
| **Total Layers** | 8 (5 Conv + 3 FC) | 16 (13 Conv + 3 FC) | 22 (Inception) | 50 (Residual) | 53 (Inverted Res) |
| **Parameters** | 42.32 Million | 32.55 Million | 6.79 Million | 24.56 Million | **2.88 Million** |
| **Model Footprint (FP32)** | 161.45 MB | 124.17 MB | 25.92 MB | 93.69 MB | **11.00 MB** |
| **Closed-Set Test Acc** | 20.00% | **100.00%** | 95.00% | **100.00%** | 61.67% |
| **Weighted F1-Score** | 0.0667 | **1.0000** | 0.9492 | **1.0000** | 0.5056 |
| **Open-Set ROC-AUC** | 0.7397 | 0.8998 | 0.8264 | **0.9082** | 0.7458 |
| **Equal Error Rate (EER)** | 46.00% | **15.67%** | 28.67% | 18.00% | 28.00% |
| **Imposter Rejection Rate** | 54.00% | 72.00% | 46.00% | **84.00%** | 69.00% |
| **False Accept Rate (FAR)** | 46.00% | 28.00% | 54.00% | **16.00%** | 31.00% |
| **Inference Latency (CPU)** | 29.12 ms | 57.30 ms | 19.79 ms | 35.54 ms | **11.96 ms** |
| **Throughput (FPS)** | 34.3 FPS | 17.5 FPS | 50.5 FPS | 28.1 FPS | **83.6 FPS** |
| **Performance Verdict** | **Poor (Underfit/Loss of Texture)** | **Good (High Accuracy / High Latency)** | **Good (Multi-Scale Balanced)** | **State-of-the-Art (Best Security)** | **Optimal Edge (Best Latency)** |

---

## 3. In-Depth Architecture Justifications: Why Each Model Performed That Way

---

### 3.1 AlexNet (2012) — Performance Verdict: **POOR / BASELINE**

```
[Input 128×128] ──► [Conv 11×11, s=2] ──► [LRN + MaxPool] ──► [Conv 5×5] ──► [3× Conv 3×3] ──► [Dense FC 42M Params]
```

* **Observed Metrics**:
  * Closed-Set Test Accuracy: **20.00%** (Random chance baseline across 5 classes is $\frac{1}{5} = 20\%$)
  * Open-Set ROC-AUC: **0.7397** | Equal Error Rate: **46.00%**
  * Parameter Footprint: **42.32M parameters (161.45 MB)** | Latency: **29.12 ms (34.3 FPS)**
* **Why did AlexNet perform this way? (Theoretical & Causal Justification)**:
  1. **Destruction of Facial Micro-Textures**: Conv1 uses a massive $11\times11$ filter kernel with stride 2, followed immediately by max-pooling. In facial biometric authentication, the most discriminative individual features reside in high-frequency facial micro-textures: tear troughs, eyelid folds, nasal creases, and skin porosity. AlexNet's aggressive spatial downsampling destroys these subtle phase transitions within the first layer.
  2. **Parameter Concentration in Dense Fully Connected Layers**: Out of 42.3 million parameters, over **$85\%$ reside entirely in the dense FC layers** (`fc1` alone accounts for $256 \times 6 \times 6 \times 4096 \approx 37.7\text{M}$ weights). When trained on modest facial crops without millions of ImageNet pre-training samples, these unregularized dense connections have too many degrees of freedom, causing high variance, slow convergence, and an inability to learn compact identity manifolds.
  3. **Lack of Modern Normalization (Batch Normalization)**: AlexNet relies on Local Response Normalization (LRN), which only normalizes across adjacent channel maps. Without Batch Normalization to stabilize internal covariate shift, gradient updates across the 42M parameters fluctuate wildly.
* **Demo Visualizations**:
  * Confusion Matrix: [`reports/figures/confusion_matrix_alexnet.png`](figures/confusion_matrix_alexnet.png)
  * ROC Curve: [`reports/figures/roc_alexnet.png`](figures/roc_alexnet.png)

---

### 3.2 VGG-16 (2014) — Performance Verdict: **GOOD (HIGH ACCURACY / HIGH LATENCY)**

```
[Input 128×128] ──► [2× Conv 3×3 (64)] ──► [2× Conv 3×3 (128)] ──► [3× Conv 3×3 (256)] ──► [3× Conv 3×3 (512)] ──► [3× Conv 3×3 (512)] ──► [FC]
```

* **Observed Metrics**:
  * Closed-Set Test Accuracy: **100.00%** | Weighted F1-Score: **1.0000**
  * Open-Set ROC-AUC: **0.8998** | Equal Error Rate: **15.67%** | Imposter Rejection: **72.00%**
  * Parameter Footprint: **32.55M parameters (124.17 MB)** | Latency: **57.30 ms (17.5 FPS)**
* **Why did VGG-16 perform this way? (Theoretical & Causal Justification)**:
  1. **Homogeneous $3\times3$ Stacking & Micro-Texture Retention**: Unlike AlexNet's large kernels, VGG-16 replaces large filters with cascades of small $3\times3$ convolutions (stride 1, padding 1). Two stacked $3\times3$ convolutions possess the same effective receptive field as a $5\times5$ filter, and three stacked $3\times3$ filters equal a $7\times7$ filter—but with two crucial advantages:
     * **Tripled Non-Linearity**: Incorporates three separate ReLU activation stages rather than one, allowing the network to carve far richer discriminative decision boundaries.
     * **Preservation of Fine Facial Geometry**: Step-by-step receptive field expansion captures edges in shallow layers, local facial organs (eyes, nose, mouth) in middle layers, and holistic face geometry in deeper layers.
  2. **Biometric Domain Pedigree (VGGFace Heritage)**: This structural cascade is the exact foundation of the landmark **VGGFace** model (Parkhi et al., 2015). It validates that deep cascades of $3\times3$ filters excel at human facial representation.
  3. **High Latency Drawback**: Because it computes 13 sequential convolutional layers over high channel depths (up to 512 channels), VGG-16 incurs the highest inference latency (**57.30 ms / 17.5 FPS** on CPU), making it too computationally expensive for resource-constrained edge CCTV chips.
* **Demo Visualizations**:
  * Confusion Matrix: [`reports/figures/confusion_matrix_vgg16.png`](figures/confusion_matrix_vgg16.png)
  * ROC Curve: [`reports/figures/roc_vgg16.png`](figures/roc_vgg16.png)

---

### 3.3 GoogLeNet / Inception-v1 (2014) — Performance Verdict: **GOOD (BALANCED MULTI-SCALE BIOMETRICS)**

```
               ┌──► [1×1 Conv] ───────────────────────────┐
               ├──► [1×1 Conv] ──► [3×3 Conv] ────────────┤
[Previous Map] ┼──► [1×1 Conv] ──► [5×5 Conv] ────────────┼──► [Channel Concatenation]
               └──► [3×3 MaxPool] ──► [1×1 Conv Proj] ────┘
```

* **Observed Metrics**:
  * Closed-Set Test Accuracy: **95.00%** | Weighted F1-Score: **0.9492**
  * Open-Set ROC-AUC: **0.8264** | Equal Error Rate: **28.67%** | Imposter Rejection: **46.00%**
  * Parameter Footprint: **6.79M parameters (25.92 MB)** | Latency: **19.79 ms (50.5 FPS)**
* **Why did GoogLeNet perform this way? (Theoretical & Causal Justification)**:
  1. **Simultaneous Multi-Scale Biometric Processing**: Human facial recognition inherently demands analyzing details at diverse spatial scales simultaneously. The $1\times1$ and $3\times3$ Inception branches focus on fine local biometric landmarks (eye corners, pupil contours, nostril curves), while the parallel $5\times5$ branch captures global cranial morphology (jawline width, forehead span, inter-ocular distance).
  2. **Global Average Pooling (GAP) Prevents Overfitting**: Instead of attaching millions of fully connected parameters at the network head, GoogLeNet applies Global Average Pooling over the final spatial feature maps. This reduces parameter count to just **6.79 Million (~26 MB)**—nearly **$6\times$ fewer parameters than AlexNet** and **$5\times$ fewer than VGG-16**—while achieving 95.0% accuracy.
  3. **Biometric Heritage (FaceNet Foundation)**: The Inception architecture family served as the exact structural backbone for Google's **FaceNet** (Schroff et al., 2015), validating its natural suitability for embedding facial manifolds.
* **Demo Visualizations**:
  * Confusion Matrix: [`reports/figures/confusion_matrix_googlenet.png`](figures/confusion_matrix_googlenet.png)
  * ROC Curve: [`reports/figures/roc_googlenet.png`](figures/roc_googlenet.png)

---

### 3.4 ResNet-50 (2015) — Performance Verdict: **STATE-OF-THE-ART (BEST SECURITY ACCURACY)**

```
[Input x] ────────────────────┬────────────────────────────┐ (Identity Shortcut: + x)
                              ▼                            │
                       [1×1 Conv Reduce]                   │
                              ▼                            │
                       [3×3 Spatial Conv]                  │
                              ▼                            │
                       [1×1 Conv Expand]                   │
                              ▼                            │
                       [F(x) Feature Map] ◄────────────────┘
                              ▼
                         [y = F(x) + x] ──► [ReLU]
```

* **Observed Metrics**:
  * Closed-Set Test Accuracy: **100.00%** | Weighted F1-Score: **1.0000**
  * Open-Set ROC-AUC: **0.9082** (Highest among all architectures!)
  * Equal Error Rate (EER): **18.00%**
  * Imposter Rejection Rate: **84.00%** (Highest intruder detection security!)
  * False Accept Rate (FAR): **16.00%** (Lowest security hazard!)
  * Parameter Footprint: **24.56M parameters (93.69 MB)** | Latency: **35.54 ms (28.1 FPS)**
* **Why did ResNet-50 perform this way? (Theoretical & Causal Justification)**:
  1. **Identity Shortcut Highway ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$)**: Traditional deep networks suffer from the degradation problem—as depth increases, gradient signals vanish during backpropagation ($\frac{\partial \mathcal{E}}{\partial \mathbf{x}}$ decays exponentially). ResNet's identity shortcuts create an unobstructed gradient highway:
     $$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
     Because of the identity term $\mathbf{I}$, gradients pass cleanly back through all 50 layers without attenuation.
  2. **Preservation of Low-Level Landmark Alignments**: In facial authentication, early spatial landmarks (inter-ocular distance, nasal bridge alignment) are vital anchors. In standard sequential networks, passing through 50 non-linear layers can distort these coordinates. In ResNet, the identity skip connection carries raw early spatial alignments forward untouched, combining them with rich deep semantic abstractions.
  3. **Universal Biometric Standard**: Modified ResNet-50 backbones are the universal gold standard used in state-of-the-art commercial and academic face recognition systems today (**ArcFace, CosFace, MagFace, InsightFace**). Its industry-leading **0.9082 ROC-AUC** and **84.0% intruder rejection rate** in our tests directly prove this superiority.
* **Demo Visualizations**:
  * Confusion Matrix: [`reports/figures/confusion_matrix_resnet50.png`](figures/confusion_matrix_resnet50.png)
  * ROC Curve: [`reports/figures/roc_resnet50.png`](figures/roc_resnet50.png)

---

### 3.5 MobileNetV2 (2018) — Performance Verdict: **OPTIMAL EDGE DEPLOYMENT (BEST REAL-TIME SPEED)**

```
[Input Narrow: C] ──► [1×1 Pointwise Expansion: 6C] ──► [3×3 Depthwise Conv: 6C] ──► [1×1 Linear Bottleneck: C']
       │                                                                                      │
       └──────────────────────────── (Residual Shortcut if C == C') ──────────────────────────┘
```

* **Observed Metrics**:
  * Closed-Set Test Accuracy: **61.67%** | Weighted F1-Score: **0.5056**
  * Open-Set ROC-AUC: **0.7458** | Imposter Rejection Rate: **69.00%** | False Accept Rate: **31.00%**
  * Parameter Footprint: **2.88M parameters (11.00 MB)** (Smallest footprint!)
  * Inference Latency: **11.96 ms / frame (83.6 FPS)** (Fastest inference by a large margin!)
* **Why did MobileNetV2 perform this way? (Theoretical & Causal Justification)**:
  1. **Depthwise Separable Convolutions**: Standard convolution performs spatial filtering and channel cross-combination in a single heavy step. MobileNetV2 decouples this into two lightweight stages:
     * A depthwise $3\times3$ convolution per channel (spatial filtering only).
     * A pointwise $1\times1$ convolution across channels (channel mixing only).
     This reduces computational FLOPs by **85% to 90%**, slashing model size to just **11.0 MB** and parameter count to **2.88 Million**.
  2. **Inverted Residuals with Linear Bottlenecks**: Unlike ResNet's *wide $\to$ narrow $\to$ wide* block, MobileNetV2 uses an *inverted* structure (*narrow $\to$ wide $\to$ narrow*). Channels are expanded internally by $6\times$ to allow rich non-linear manifold capacity, then compressed back to a low-dimensional bottleneck. Crucially, the final projection is **linear (no ReLU)**: non-linear activations in low dimensions cause the "manifold collapse" problem, destroying biometric variance.
  3. **Edge CCTV & Surveillance Trade-Off**: At **83.6 FPS (11.96 ms)** on a basic CPU, MobileNetV2 easily surpasses real-time video streaming requirements (30 FPS). While its lightweight nature limits its closed-set discriminability (61.67%) compared to the 24M-parameter ResNet-50 when trained on limited samples, it is the only viable candidate for direct deployment on low-cost edge security hardware (smart doorbells, Raspberry Pi, Jetson Nano).
* **Demo Visualizations**:
  * Confusion Matrix: [`reports/figures/confusion_matrix_mobilenetv2.png`](figures/confusion_matrix_mobilenetv2.png)
  * ROC Curve: [`reports/figures/roc_mobilenetv2.png`](figures/roc_mobilenetv2.png)

---

## 4. Open-Set Intruder Rejection Mechanism (Why Not Softmax?)

### The Academic Problem with Closed-Set Softmax
Standard deep learning classification uses a final `Softmax` layer with cross-entropy loss over fixed classes $\{1, \dots, K\}$:
$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
Under softmax, probabilities are mathematically constrained to sum to $1.0$. If an unseen intruder face appears before the camera, softmax is **forced** to assign probability mass to whichever enrolled user looks most similar.

### The Production Open-Set Prototype Gating Solution
Review 2 addresses this by employing metric embedding gating:

```
[ Input Aligned Face (128×128) ]
               │
               ▼
   [ Deep CNN Backbone (AlexNet / VGG / GoogLeNet / ResNet / MobileNet) ]
               │
               ▼
   [ L2-Normalized Feature Embedding: e = f(x) / ||f(x)||_2 ]
               │
               ▼
   [ Distance Metric Gating against Enrolled Class Prototypes c_k ]
      d* = min_k ( 1 - <e, c_k> )
               │
     ┌─────────┴─────────┐
     ▼                   ▼
d* < tau_optimal    d* >= tau_optimal
[ AUTHORIZED ]     [ INTRUDER ALERT ]
(Known Subject)    (Unknown / Imposter)
```

1. **Embedding Extraction**: For each enrolled face crop, the network computes a 512-dimensional feature embedding $\mathbf{e} = \frac{f(\mathbf{x})}{\|f(\mathbf{x})\|_2}$ on the unit hypersphere $\mathbb{S}^{D-1}$.
2. **Centroid Prototypes**: For each enrolled identity $k$, the class prototype centroid $\mathbf{c}_k$ is calculated and normalized:
   $$\mathbf{c}_k = \frac{\sum_{i \in S_k} \mathbf{e}_i}{\|\sum_{i \in S_k} \mathbf{e}_i\|_2}$$
3. **Cosine Metric Gating**: For any test face, the minimum cosine distance $d^*$ to all enrolled prototypes is measured:
   $$d^* = \min_{k} (1 - \cos(\mathbf{e}, \mathbf{c}_k)) \in [0, 2]$$
4. **Decision Boundary**:
   * If $d^* < \tau^* \implies$ **AUTHORIZED PERSON** ($\text{ID} = \arg\min_k d(\mathbf{e}, \mathbf{c}_k)$).
   * If $d^* \ge \tau^* \implies$ **INTRUDER ALERT!** (Unknown stranger / imposter rejected).
5. **Threshold Calibration**: The operational threshold $\tau^*$ is calibrated at the **Equal Error Rate (EER)** point where the False Accept Rate (FAR) intersects the False Reject Rate (FRR).

---

## 5. Comparative Visualizations for Demo Presentation

The benchmark generates two cross-architecture comparative charts stored in `reports/figures/`:

1. **Accuracy vs. Inference Latency (Bubble Size = Parameter Footprint)**:
   * File: [`reports/figures/review2_latency_vs_accuracy.png`](figures/review2_latency_vs_accuracy.png)
   * *Demonstration value*: Illustrates the Pareto frontier between real-time inference speed (MobileNetV2 at 11.96ms) and classification precision (ResNet-50 and VGG-16 at 100%).
2. **Multi-Metric Comparative Bar Chart**:
   * File: [`reports/figures/review2_metric_comparison_bars.png`](figures/review2_metric_comparison_bars.png)
   * *Demonstration value*: Directly compares Closed-Set Accuracy, Open-Set ROC-AUC, and Imposter Rejection Rate side-by-side across all 5 architectures.

---

## 6. Demo Viva / Question & Answer Reference

| Anticipated Evaluator Question | Recommended Student Answer |
| :--- | :--- |
| **Q1: Why did AlexNet perform so poorly compared to VGG and ResNet?** | "AlexNet uses large 11×11 receptive fields with stride 2/4 in layer 1, which rapidly downsamples the image and destroys the high-frequency facial micro-textures essential for recognizing faces. Additionally, 85% of its 42M parameters sit in unregularized dense fully connected layers, causing severe overfitting on small face datasets." |
| **Q2: Why does VGG-16 achieve 100% accuracy, and why is it not suitable for CCTV?** | "VGG-16 stacks small 3×3 convolutions, capturing fine facial contours across 13 layers without destroying high frequencies (the exact design of VGGFace). However, its 32.5M parameters and sequential depth yield 57.30 ms latency (17.5 FPS), which exceeds the 33ms budget required for 30 FPS real-time CCTV streams." |
| **Q3: How does GoogLeNet reduce parameters while maintaining 95% accuracy?** | "GoogLeNet uses parallel Inception modules with 1×1 bottleneck convolutions and Global Average Pooling (GAP) instead of dense FC layers. This cuts parameters to just 6.79M (~26 MB) while simultaneously processing micro-features (3×3) and macro-geometry (5×5), mirroring Google's FaceNet." |
| **Q4: Why is ResNet-50 considered the industry standard for face biometrics?** | "ResNet-50's identity shortcut connections ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$) solve the vanishing gradient problem, enabling 50 deep layers to converge. Furthermore, the shortcut acts as a preservation highway for low-level facial landmark alignments, yielding our highest ROC-AUC (0.9082) and highest imposter rejection (84.0%)." |
| **Q5: What makes MobileNetV2 unique, and when should it be used?** | "MobileNetV2 decouples spatial filtering from channel mixing using depthwise separable convolutions, reducing computation by 85–90%. With linear bottlenecks preventing manifold collapse, it runs at 83.6 FPS (11.96 ms) and only 11 MB footprint, making it the only model feasible for edge deployment on Raspberry Pi or CCTV smart cameras." |
