# Architecture Justification: GoogLeNet / Inception-v1 (2014)

> **Quick Summary:** GoogLeNet achieved 83.3% accuracy while being 4x smaller and 4x faster than VGG-16. Its breakthrough idea is the "Inception Module," which looks at a face at multiple scales ($1\times1, 3\times3, 5\times5$) simultaneously.

---

## 1. Key Numbers to Remember for Review

* **Closed-Set Test Accuracy:** **83.3%**
* **Intruder Rejection Rate:** **77.5%**
* **Open-Set ROC-AUC:** **0.883**
* **Inference Speed:** **8.8 ms per frame (114 FPS — extremely fast)**
* **Parameters / Size:** **7.17 Million parameters (27.3 MB)**
* **Verdict:** **High Speed, Balanced Accuracy, and Pedigree of Google's FaceNet**

---

## 2. Review Speaking Script (What to say to the Examiner in 45 Seconds)

> *"Sir/Ma'am, our third architecture is **GoogLeNet (Inception-v1)**, which belongs to the exact architectural family used in **Google's FaceNet**.*  
>  
> *GoogLeNet achieved **83.3% test accuracy** matching VGG-16, but runs at a blazing fast **8.8 milliseconds per frame (114 FPS)** with only **7.17 million parameters**.*  
>  
> *The reason it performs so efficiently is its **multi-scale Inception design**. In every block, it processes the image through four parallel branches at the same time: 1x1 filters for local features, 3x3 filters for eyes and nose, and 5x5 filters for overall head geometry. Then it concatenates all of them.*  
>  
> *To keep computations low, it uses smart **1x1 bottleneck convolutions** before the expensive filters and replaces heavy dense layers with **Global Average Pooling**. This prevents overfitting and makes GoogLeNet lightweight, fast, and accurate."*

---

## 3. High-Level Code Explanation (Simple & Easy to Manage)

The model code is in [`src/models/googlenet.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/googlenet.py), and the runner script is [`scripts/run_googlenet.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_googlenet.py).

Here is how the code works in 5 simple steps:

1. **Step 1: Input Face Crop**  
   * The input is a $128 \times 128 \times 3$ RGB cropped face image.
2. **Step 2: Stem Convolution (`self.conv1`, `conv2`, `conv3`)**  
   * An initial $7 \times 7$ convolution (stride 2) and max-pooling quickly extract early spatial lines.
3. **Step 3: Multi-Scale Inception Blocks (`InceptionBlock` class in code)**  
   * Every Inception block runs 4 parallel paths:
     - **Branch 1:** $1 \times 1$ Conv (fine pixel variations)
     - **Branch 2:** $1 \times 1 \to 3 \times 3$ Conv (facial components: eyes, nose)
     - **Branch 3:** $1 \times 1 \to 5 \times 5$ Conv (global face shape)
     - **Branch 4:** $3 \times 3$ MaxPool $\to 1 \times 1$ Conv
   * All 4 branch outputs are combined into one channel stack via `torch.cat(...)`.
4. **Step 4: Global Average Pooling (`self.avgpool` in code)**  
   * Instead of flattening millions of values into a huge dense layer, it averages each channel over the spatial area, compressing the map into a 1024-D vector.
5. **Step 5: 512-D Embedding & Intruder Check (`self.fc` & `OpenSetGatingEngine`)**  
   * A single linear layer maps $1024 \to 512$ with Dropout (0.4) to produce the normalized face vector.  
   * Prototype distance check classifies strangers when distance is higher than $\tau^* = 0.211$.

---

## 4. Why Did GoogLeNet Get 83.3% Accuracy & High Speed? (Simple Justification)

| Factor | What Happened | Why It Affected Accuracy & Speed |
| :--- | :--- | :--- |
| **Multi-Scale Inception** | 4 parallel filter sizes in every block. | Recognizes small details (iris, lips) and big landmarks (jawline, forehead) at the same time. |
| **$1\times1$ Bottlenecks** | Reduces channel dimensions before $3\times3$ and $5\times5$ filters. | Cuts parameters down to just 7.17M, preventing memorization on small face datasets. |
| **Global Average Pooling** | Averages each channel instead of using huge fully connected layers. | Keeps model size to just 27 MB and boosts inference to 114 FPS. |

---

## 5. How to Run the Demo in Viva

Run this single command in terminal (loads saved checkpoint instantly):

```bash
python3 scripts/run_googlenet.py --test-only
```
