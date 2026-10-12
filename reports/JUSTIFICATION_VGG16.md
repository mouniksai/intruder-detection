# Architecture Justification: VGG-16 (2014)

> **Quick Summary:** VGG-16 achieved high face recognition accuracy (83.3%) because it replaced big filters with small, stacked 3x3 filters that capture fine facial textures. However, it is the slowest model in our suite (44.9 ms) and requires heavy compute power.

---

## 1. Key Numbers to Remember for Review

* **Closed-Set Test Accuracy:** **83.3%**
* **Intruder Rejection Rate:** **85.0%**
* **Open-Set ROC-AUC:** **0.890**
* **Inference Speed:** **44.9 ms per frame (~22 FPS — slowest in the suite)**
* **Parameters / Size:** **32.6 Million parameters (124.2 MB)**
* **Verdict:** **High Accuracy & Strong Biometrics, but Heavy & Slow for Edge Devices**

---

## 2. Review Speaking Script (What to say to the Examiner in 45 Seconds)

> *"Sir/Ma'am, for our second architecture, we implemented **VGG-16 (2014)**, which is the foundational design behind Oxford's famous **VGGFace** system.*  
>  
> *VGG-16 achieved **83.3% test accuracy** and **85.0% intruder rejection**, which is a significant improvement over AlexNet.*  
>  
> *The reason it performs well is its design: instead of large filters, VGG-16 stacks small, uniform **3x3 filters** throughout 16 layers. Two 3x3 filters have the same receptive field as a 5x5 filter, but with twice the non-linear activations. This allows VGG-16 to preserve fine facial micro-textures like skin pores, wrinkles, and eye shapes.*  
>  
> *However, its drawback is computational cost: stacking 512-channel convolutional layers requires heavy computations, making it our slowest architecture at **44.9 milliseconds per frame**, which is not ideal for low-cost CCTV hardware."*

---

## 3. High-Level Code Explanation (Simple & Easy to Manage)

The model code is in [`src/models/vgg16.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/vgg16.py), and the runner script is [`scripts/run_vgg16.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_vgg16.py).

Here is the code structure in 5 simple steps:

1. **Step 1: Input Face Crop**  
   * The input is a $128 \times 128 \times 3$ RGB cropped face image.
2. **Step 2: Deep 3x3 Stacking (`self.features` in code)**  
   * Instead of big filters, it passes the face through 5 blocks of $3 \times 3$ convolutions with Batch Normalization (channels grow: $64 \to 128 \to 256 \to 512 \to 512$).  
   * Max-pooling cuts the image dimensions in half between blocks.
3. **Step 3: Adaptive Pooling (`self.avgpool` in code)**  
   * Resizes the spatial output to a fixed $4 \times 4$ grid ($512 \times 4 \times 4 = 8,192$ values).
4. **Step 4: 512-D Face Embedding (`self.classifier_block` in code)**  
   * Two dense layers ($8192 \to 2048 \to 512$) project the features into our standard 512-D biometric face vector with Dropout (0.5) to avoid overfitting.
5. **Step 5: Output & Intruder Check (`self.head` & `OpenSetGatingEngine`)**  
   * Linear head classifies the face into one of the 30 registered people.  
   * Prototype cosine distance check rejects unknown faces when distance exceeds $\tau^* = 0.036$.

---

## 4. Why Did VGG-16 Get 83.3% Accuracy? (Simple Justification)

| Factor | What Happened | Why It Affected Accuracy |
| :--- | :--- | :--- |
| **All $3\times3$ Convolutions** | Replaced rough big filters with cascades of small filters. | Captures crisp, high-frequency facial textures (eyes, nose, mouth contours). |
| **Deeper Network (16 Layers)** | 16 layers with ReLU activations. | Builds better combinations of low-level lines into full facial profiles. |
| **Batch Normalization (BN)** | Normalizes activations after every layer. | Stabilizes learning across all 30 face classes. |
| **Why is it slow?** | 512 channels repeated across late layers with no bottleneck reductions. | High FLOP count makes per-frame latency 44.9 ms. |

---

## 5. How to Run the Demo in Viva

Run this single command in terminal (loads saved checkpoint instantly):

```bash
python3 scripts/run_vgg16.py --test-only
```
