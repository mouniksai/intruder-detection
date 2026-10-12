# Architecture Justification: ResNet-50 (2015)

> **Quick Summary:** ResNet-50 achieved the **highest performance in our entire review suite** (**87.5% accuracy, 0.924 ROC-AUC, 85% intruder rejection**). Its revolutionary "skip connections" allow the 50-layer network to learn without vanishing gradients, making it the universal gold standard for modern face recognition.

---

## 1. Key Numbers to Remember for Review

* **Closed-Set Test Accuracy:** **87.5% (Highest overall)**
* **Macro F1-Score:** **0.877 (Highest overall)**
* **Open-Set ROC-AUC:** **0.924 (Highest overall)**
* **Intruder Rejection Rate:** **85.0%**
* **Inference Speed:** **13.6 ms per frame (73.5 FPS — real-time)**
* **Parameters / Size:** **24.57 Million parameters (93.7 MB)**
* **Verdict:** **Best Overall Architecture (The Biometric Gold Standard used by ArcFace & InsightFace)**

---

## 2. Review Speaking Script (What to say to the Examiner in 45 Seconds)

> *"Sir/Ma'am, for our selected deep architecture, we evaluated **ResNet-50 (2015)**, which is the universal gold standard backbone used in modern face recognition systems like **ArcFace** and **InsightFace**.*  
>  
> *ResNet-50 produced our **best overall results: 87.5% test accuracy, 0.877 F1-score, and an outstanding 0.924 open-set ROC-AUC**, running at 73.5 frames per second.*  
>  
> *Why did ResNet-50 perform so well? Because of its famous **residual skip connections**: $y = F(x) + x$. In traditional networks, stacking 50 layers causes gradients to vanish and degrades feature quality. ResNet adds a direct bypass highway around convolutional blocks.*  
>  
> *This bypass lets low-level facial landmark positions (eye corners, jawline, nose bridge) travel directly into deep layers without being distorted. The result is extremely tight clustering for the same person and wide separation for strangers and intruders."*

---

## 3. High-Level Code Explanation (Simple & Easy to Manage)

The model code is in [`src/models/resnet50.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/resnet50.py), and the runner script is [`scripts/run_resnet50.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_resnet50.py).

Here is the code structure in 5 simple steps:

1. **Step 1: Input Face Crop**  
   * The input is a $128 \times 128 \times 3$ RGB cropped face image.
2. **Step 2: Stem Convolution (`self.conv1`, `bn1`, `maxpool`)**  
   * A $7 \times 7$ convolution with stride 2 and max-pooling extracts initial edges and contours.
3. **Step 3: Bottleneck Residual Blocks (`BottleneckBlock` class in code)**  
   * Organizes 50 layers into 4 residual stages containing `[3, 4, 6, 3]` bottleneck blocks.  
   * Each block uses a $1\times1 \to 3\times3 \to 1\times1$ sandwich to reduce and restore channel depth.  
   * **The Key Step:** `out = self.relu(identity + out)` adds the original input directly to the processed output.
4. **Step 4: Global Pooling & 512-D Face Embedding (`self.avgpool` & `self.fc_embed`)**  
   * Averages spatial dimensions down to $2048 \times 1 \times 1$, then projects through `Linear(2048, 512)` and Batch Normalization.  
   * Normalizes the vector onto the unit hypersphere with $\ell_2$ normalization (`F.normalize(x, p=2, dim=1)`).
5. **Step 5: Classification Head & Intruder Check (`self.classifier` & `OpenSetGatingEngine`)**  
   * Multi-class head classifies the face into one of the 30 registered people.  
   * Prototype cosine distance check rejects unknown intruders when distance is greater than $\tau^* = 0.235$.

---

## 4. Why Did ResNet-50 Achieve the Highest Accuracy? (Simple Justification)

| Factor | What Happened | Why It Resulted in 87.5% Accuracy |
| :--- | :--- | :--- |
| **Residual Shortcut Connections** | Input $x$ is added directly to output $F(x)$. | Creates a gradient highway; all 50 layers learn without gradient vanishing. |
| **Landmark Anchor Preservation** | Spatial landmarks pass through the shortcuts without being degraded by non-linearities. | Keeps exact biometric measurements (eye-to-eye distance, nose-to-chin ratio) intact. |
| **Clean Embedding Separation** | 512-D $\ell_2$-normalized output space. | Groups enrolled identities into compact clusters and separates unfamiliar intruders with a 0.924 ROC-AUC. |

---

## 5. How to Run the Demo in Viva

Run this single command in terminal (loads saved checkpoint instantly):

```bash
python3 scripts/run_resnet50.py --test-only
```
