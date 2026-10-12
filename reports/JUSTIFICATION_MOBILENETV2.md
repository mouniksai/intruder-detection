# Architecture Justification: MobileNetV2 (2018)

> **Quick Summary:** MobileNetV2 is our **standout edge architecture**. It achieved **83.3% test accuracy and an outstanding 0.913 ROC-AUC** while using only **2.9 Million parameters (11.0 MB)**. It is by far the most practical model for real-time CCTV security cameras and Raspberry Pi deployment.

---

## 1. Key Numbers to Remember for Review

* **Closed-Set Test Accuracy:** **83.3%**
* **Macro F1-Score:** **0.839**
* **Open-Set ROC-AUC:** **0.913 (2nd highest, nearly matching ResNet-50)**
* **Intruder Rejection Rate:** **82.5%**
* **Inference Speed:** **17.5 ms per frame (57.1 FPS — real-time)**
* **Parameters / Size:** **2.90 Million parameters (ONLY 11.0 MB — 88% smaller than ResNet-50)**
* **Verdict:** **Best Architecture for Edge CCTV Cameras & Embedded IoT Security Systems**

---

## 2. Review Speaking Script (What to say to the Examiner in 45 Seconds)

> *"Sir/Ma'am, for our lightweight edge architecture, we selected **MobileNetV2 (2018)**.*  
>  
> *MobileNetV2 achieved **83.3% test accuracy and a high 0.913 open-set ROC-AUC**, while having a footprint of **just 2.9 million parameters and 11 megabytes**.*  
>  
> *How does it achieve such high biometric accuracy with such a tiny size? It uses three smart innovations:*  
> *1. **Depthwise Separable Convolutions:** It splits standard convolution into spatial filtering per channel and 1x1 channel mixing. This cuts FLOPs and mathematical operations by nearly 90%.*  
> *2. **Inverted Residuals:** Instead of compressing channels first, it temporarily expands channels to let the depthwise filter learn rich features, then projects back down.*  
> *3. **Linear Bottlenecks:** It removes the ReLU non-linearity at the bottleneck projection. This prevents valuable biometric information from being zeroed out in low-dimensional spaces.*  
>  
> *Therefore, MobileNetV2 is our top recommendation for real-world deployment on CCTV security cameras and edge devices like Raspberry Pi or Jetson Nano."*

---

## 3. High-Level Code Explanation (Simple & Easy to Manage)

The model code is in [`src/models/mobilenetv2.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/mobilenetv2.py), and the runner script is [`scripts/run_mobilenetv2.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_mobilenetv2.py).

Here is the code structure in 5 simple steps:

1. **Step 1: Input Face Crop**  
   * The input is a $128 \times 128 \times 3$ RGB cropped face image.
2. **Step 2: Stem Convolution (`self.features[0]`)**  
   * Initial standard $3 \times 3$ convolution with 32 filters (stride 2) for early edge detection.
3. **Step 3: Inverted Residual Blocks (`InvertedResidual` class in code)**  
   * Every block executes 3 consecutive sub-steps:
     - **Expansion:** $1 \times 1$ Conv expands channels by factor 6.
     - **Depthwise Filtering:** $3 \times 3$ Conv with `groups=hidden_dim` filters each channel separately (saving 88% FLOPs).
     - **Linear Bottleneck:** $1 \times 1$ Conv compresses channels back with **no ReLU** to preserve biometric features.
   * If input and output shapes match, it adds a residual skip: `x + self.conv(x)`.
4. **Step 4: Global Pooling & 512-D Face Embedding (`self.classifier` in code)**  
   * Adaptive pooling compresses features to $1 \times 1$, followed by a Linear projection ($1280 \to 512$) to form the 512-D biometric face vector.
5. **Step 5: Output & Intruder Check (`OpenSetGatingEngine`)**  
   * Prototype distance check flags unknown strangers as intruders when distance exceeds $\tau^* = 0.283$.

---

## 4. Why Did MobileNetV2 Get 83.3% Accuracy & 0.913 ROC-AUC? (Simple Justification)

| Factor | What Happened | Why It Resulted in High Accuracy with 11 MB Size |
| :--- | :--- | :--- |
| **Depthwise Separable Convolutions** | Separates spatial filtering from channel mixing. | Cuts parameters to 2.9M while keeping feature representation sharp. |
| **Linear Bottlenecks** | No ReLU activation after the final $1\times1$ projection. | Prevents non-linear destruction of subtle biometric signals in compact dimensions. |
| **Inverted Residual Connections** | Memory-efficient residual shortcuts across bottlenecks. | Maintains gradient flow and stabilizes fine-tuning on our 30 enrolled subjects. |

---

## 5. How to Run the Demo in Viva

Run this single command in terminal (loads saved checkpoint instantly):

```bash
python3 scripts/run_mobilenetv2.py --test-only
```
