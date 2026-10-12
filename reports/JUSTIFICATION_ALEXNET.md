# Architecture Justification: AlexNet (2012)

> **Quick Summary:** AlexNet was our baseline model. It is fast, but achieved the lowest accuracy among all models because its early filters are too big and its network is too shallow for subtle face details.

---

## 1. Key Numbers to Remember for Review

* **Closed-Set Test Accuracy:** **77.8%**
* **Intruder Rejection Rate:** **77.5%**
* **Open-Set ROC-AUC:** **0.830**
* **Inference Speed:** **6.6 ms per frame (over 150 FPS — very fast)**
* **Parameters / Size:** **42.3 Million parameters (161.5 MB)**
* **Verdict:** **Baseline Model (Average Accuracy, High Memory Footprint)**

---

## 2. Review Speaking Script (What to say to the Examiner in 45 Seconds)

> *"Good morning / afternoon sir/ma'am. We evaluated **AlexNet** as our foundational 2012 baseline architecture.*  
>  
> *AlexNet gave us **77.8% test accuracy** and **77.5% intruder rejection**. While it is computationally fast during inference (taking only 6.6 milliseconds), it performed the lowest among all 5 models.*  
>  
> *The reason is simple: AlexNet uses large 11x11 filters in its very first layer, which blurs out fine facial details like eye corners and nose contours. Furthermore, it only has 5 convolutional layers, so it cannot learn deep, pose-invariant face features. Finally, most of its 42 million parameters are packed into dense layers, which makes it heavy in storage (161 MB) without giving high biometric accuracy."*

---

## 3. High-Level Code Explanation (Simple & Easy to Manage)

The model code is in [`src/models/alexnet.py`](file:///Users/mouniksai/Downloads/intruder-detection/src/models/alexnet.py), and the runner script is [`scripts/run_alexnet.py`](file:///Users/mouniksai/Downloads/intruder-detection/scripts/run_alexnet.py).

Here is the entire model pipeline in 5 simple steps:

1. **Step 1: Input Face Crop**  
   * The input is a $128 \times 128 \times 3$ RGB cropped face image.
2. **Step 2: Convolutional Features (`self.features` in code)**  
   * Layer 1 uses a big $11 \times 11$ filter to pick up basic edges and colors.  
   * Followed by $5 \times 5$ and three $3 \times 3$ convolutions to extract shape outlines.
3. **Step 3: Adaptive Pooling (`self.avgpool` in code)**  
   * Compresses the spatial map down to a fixed $6 \times 6$ size.
4. **Step 4: 512-D Face Embedding (`self.fc2` in code)**  
   * A dense layer converts the pooled features into a 512-dimensional vector. This 512-D vector acts as the unique digital "fingerprint" of the face.
5. **Step 5: Classification Head & Intruder Check (`self.classifier` & `OpenSetGatingEngine`)**  
   * `self.classifier` classifies the face into one of the 30 enrolled people.  
   * For strangers, `OpenSetGatingEngine` measures the cosine distance to all stored face prototypes. If the distance is above threshold $\tau^* = 0.109$, it flags the person as an **INTRUDER**.

---

## 4. Why Did AlexNet Get 77.8% Accuracy? (Simple Justification)

| Factor | What Happened | Why It Affected Accuracy |
| :--- | :--- | :--- |
| **Big $11\times11$ Filters** | The first layer takes huge jumps across pixels. | It misses tiny facial textures (eyebrow hairs, eye corners, lip edges). |
| **Shallow Depth (Only 5 Conv Layers)** | Network is only 8 layers deep overall. | Cannot build advanced facial geometry (e.g., recognizing someone when their head turns). |
| **Heavy Fully Connected Layers** | Over 90% of parameters are in dense linear layers. | Prone to memorizing images rather than learning generalizable face features. |

---

## 5. How to Run the Demo in Viva

Run this single command in terminal (takes just 3 seconds, loads checkpoint without retraining):

```bash
python3 scripts/run_alexnet.py --test-only
```
