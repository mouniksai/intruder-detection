# Computer Vision Case Study — Intruder Detection via Face Recognition
### Full Roadmap: Review 1 (Classical Feature Extraction + Classifiers) → Review 2 (Deep Learning)

> Team size: any number — each member owns one feature extractor / role; scale the lists below up or down to match your headcount.
> Scope: Face detection/recognition system that identifies a set of known people and flags anyone else as an "intruder/unknown", with an automatic alert.

---

## 0. Problem Framing (read this first)

This is **not** plain multi-class classification. It is **open-set recognition**:

- N known identities (the team members enrolled) → system must say "Person X"
- Anyone else → system must say **"Unknown / Intruder"**, including people it has never seen before

This distinction must be baked in from the start — it affects dataset design, evaluation metrics, and thresholding logic in both reviews. Do not retrofit it later.

---

## 1. Team Workflow Setup (before writing any feature/model code)

- [ ] Create a shared repo (GitHub/GitLab) with one shared `requirements.txt`
- [ ] Agree on a **common data contract** so the team can work in parallel without merge conflicts:
  ```python
  extract_features_<name>(face_image: np.ndarray) -> np.ndarray   # fixed-length vector
  ```
  Every feature extractor obeys this signature. The classifier/evaluation script never needs to know *how* the vector was produced.
- [ ] Set up the folder skeleton:
  ```
  /data/raw/<person_name>/*.jpg
  /data/processed/<person_name>/*.jpg      # after detection + alignment + resize
  /src/preprocessing/detect_align.py       # shared, written once
  /src/features/{pca,lda,lbp,hog,gabor}.py # one file per person
  /src/classifiers/train_eval.py           # shared evaluation harness
  /src/deep/                               # Review 2
  /notebooks/                              # exploration, plots
  /reports/                                # review write-ups
  ```
- [ ] Use one notebook per person for experimentation, but push final logic into `.py` modules — reusable code reads better to evaluators than notebook-only work.

---

## 2. Dataset Strategy (the most important decision — lock this in early)

### 2.1 Build your own primary dataset
- [ ] Each team member captures **60–100 images of themselves** via a webcam capture script (grab a frame every ~0.5s while moving your head), varying:
  - pose (left/right/up/down)
  - lighting (bright/dim/backlit)
  - expression
  - with/without glasses, hair up/down
- [ ] Collect **~50–100 "unknown" faces** — people outside the enrolled team, public dataset images, stock photos — labeled `unknown`. Without this pool, the system can *never* be tested on the actual "intruder" case, since it will only ever have seen the enrolled classes and will always force-fit into one of them.
- [ ] Target totals: **one known class per enrolled member (~60–100 images each)** + **1 unknown pool (100+ images, held out from training)**. Totals scale linearly with however many people you enroll.

### 2.2 Supplement with a public dataset (for a generalization experiment)
- [ ] **AT&T/ORL Faces** — 40 subjects × 10 images, controlled, grayscale. Good for classical-feature sanity checks.
- [ ] **LFW (Labeled Faces in the Wild)** subset — messier, real-world lighting/pose. Good for stress-testing Review 2's deep models.
- [ ] Report structure: *"Primary results on our self-collected dataset; secondary generalization experiment on AT&T/LFW."* This one addition noticeably raises the rigor of the report.

### 2.3 Splits
- [ ] Stratified train/val/test per known identity (e.g., 70/15/15)
- [ ] Keep a **pure unknown test set** never touched during training — used only to measure false-accept rate

### 2.4 Augmentation (cheap accuracy + robustness win — use in both reviews)
- [ ] Small rotations (±15°), horizontal flip, brightness/contrast jitter, Gaussian noise — via OpenCV / `imgaug` / `albumentations`
- [ ] Apply augmentation **before** feature extraction, not after — note this explicitly in the report

---

## 3. Shared Preprocessing Pipeline (build once, everyone imports it)

1. [ ] **Face detection** — OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) for simplicity, or MTCNN (`mtcnn` package) for better accuracy. Worth a small side-comparison of both in the report.
2. [ ] **Alignment** — detect eye landmarks (dlib 68-point or MTCNN 5-point), rotate so eyes are horizontal. This measurably boosts HOG/LBP/PCA performance since they're all sensitive to misalignment.
3. [ ] **Normalization** — crop to bounding box + margin, resize to a fixed size (e.g., 128×128), convert to grayscale for classical features (keep color for deep learning), apply CLAHE histogram equalization to reduce lighting variance.
4. [ ] Save processed faces to `/data/processed/` so every feature-extractor script reads from the **same clean input** — required for a fair comparison across methods.

---

## 4. Review 1 — Classical Feature Extraction + Classifiers

Assign **one extractor per team member**. If your team is bigger than the core list below, add extractors from the "extended options" list; if smaller, combine two extractors per person or drop the least critical one. Each write-up should include: **theory summary, implementation, feature vector dimensionality, and results.**

### Core options (pick as many as you have members for, in priority order)

| # | Feature Extractor | Core Idea | Library |
|---|---|---|---|
| 1 | **Eigenfaces (PCA)** | Project flattened pixels onto top-k principal components capturing max variance | `sklearn.decomposition.PCA` |
| 2 | **Fisherfaces (LDA)** | Like PCA but supervised — maximizes between-class vs within-class scatter | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` |
| 3 | **LBP (Local Binary Patterns)** | Per-pixel texture code from thresholded neighborhood comparisons, summarized as a histogram per region | `skimage.feature.local_binary_pattern` |
| 4 | **HOG (Histogram of Oriented Gradients)** | Gradient orientation histograms over cells/blocks — captures edge/shape structure | `skimage.feature.hog` |
| 5 | **Gabor filter bank** | Bank of orientation/frequency-selective filters convolved with the face, pooled per filter | `cv2.getGaborKernel` |

### Extended options (use if your team has more than 5 people, or want a substitute)

| # | Feature Extractor | Core Idea | Library |
|---|---|---|---|
| 6 | **SIFT/ORB + Bag-of-Visual-Words** | Detect scale/rotation-invariant keypoints, cluster descriptors into a visual vocabulary, represent each face as a histogram over that vocabulary | `cv2.SIFT_create` / `cv2.ORB_create` |
| 7 | **Haar-like features (Viola-Jones style)** | Rectangular intensity-difference filters at multiple scales/positions, the same primitive used for face *detection*, repurposed here for recognition features | `cv2.CascadeClassifier` / custom |
| 8 | **DCT / Wavelet transform features** | Frequency-domain decomposition (Discrete Cosine or Wavelet Transform), keep low-frequency coefficients as a compact descriptor | `scipy.fft`, `PyWavelets` |
| 9 | **Color/intensity histogram + moments** | Global or block-wise pixel intensity statistics (mean, variance, skewness) — a simple baseline to contrast against texture/shape features | `numpy`, `scipy.stats` |
| 10 | **Local Phase Quantization (LPQ)** | Texture descriptor robust to blur, based on quantized phase of local Fourier transforms | custom / `mahotas` |
| 11 | **Zernike moments** | Shape descriptors invariant to rotation, computed from orthogonal Zernike polynomials over the face region | `mahotas.features.zernike_moments` |

### For every extractor, each member should:
- [ ] Feed the extracted vector into **the same classifier(s)** for a controlled comparison: **SVM (linear + RBF)**, **KNN**, optionally Random Forest / Logistic Regression
- [ ] Run **k-fold cross-validation** — report mean ± std accuracy, not a single number
- [ ] Report **precision, recall, F1, confusion matrix** per class — accuracy alone hides intruder-detection failures (e.g. 95% accuracy while every "unknown" is still misclassified as a known person)
- [ ] Tune classifier hyperparameters with `GridSearchCV`
- [ ] Implement **open-set thresholding**: if classifier confidence / nearest-neighbor distance exceeds a threshold, output "Unknown" instead of forcing a class. Tune the threshold on validation data and report an **ROC curve** for known-vs-unknown separation

### Extra additions that raise the ceiling of Review 1
- [ ] **Dimensionality reduction before classification** for high-dim features (HOG/Gabor can be thousands of dims) — apply PCA as a second stage, plot accuracy vs. #components
- [ ] **Feature fusion** — concatenate 2–3 of the chosen features (e.g. HOG + LBP), show whether fusion beats any single feature. Ties the whole team's individual work into one shared experiment
- [ ] **t-SNE/PCA 2D visualization** of each feature space colored by identity (`sklearn.manifold.TSNE`) — visually strong for the report/presentation, cheap to produce
- [ ] **Robustness mini-study** — test each feature's accuracy under synthetic occlusion (black rectangle over eyes/mouth) or lighting shift
- [ ] **Timing/efficiency comparison** — extraction time + classification time per feature, foreshadowing the real-time/deployment discussion in Review 2

---

## 5. Review 2 — Deep Learning

### 5.1 Two parallel tracks (deliberately mirrors Review 1's structure)
1. [ ] **CNN from scratch** — small architecture (3–4 conv blocks + FC), trained directly on your face crops. Will likely *underperform* transfer learning given your small dataset — that gap itself is a good discussion point.
2. [ ] **Transfer learning / pretrained embeddings** — use a pretrained face embedding network (FaceNet, ArcFace via `deepface` or `insightface`, or a pretrained ResNet/MobileNet backbone) to get a fixed-length embedding per face, then — exactly like Review 1 — feed it into a classifier (SVM/KNN) OR fine-tune the last few layers end-to-end.

### 5.2 Open-set recognition, properly
- [ ] With embeddings, use **cosine/Euclidean distance to enrolled prototypes** (not softmax over fixed classes) — this is the actual production approach (how Face ID-style systems work), and supports adding/removing a person without retraining
- [ ] Tune the distance threshold on a validation set combining known + held-out unknown pool
- [ ] Report **ROC/AUC** for the known-vs-unknown decision, and **False Accept Rate vs. False Reject Rate** at different thresholds — the security-relevant metrics for an actual intruder system (a false accept is far worse than a false reject here)

### 5.3 Extras that add real depth to Review 2
- [ ] **Data augmentation via a training pipeline** (`torchvision.transforms` / `tf.keras` layers) — random crop, flip, color jitter, cutout; compare accuracy with vs. without
- [ ] **Explainability** — Grad-CAM heatmaps on the CNN showing which facial regions drove a decision
- [ ] **Unified model comparison table**: classical-best vs. CNN-from-scratch vs. transfer-learning, across accuracy, FAR/FRR, inference time, model size — this single table ties both reviews together
- [ ] **Robustness testing carried over from Review 1** (occlusion, lighting, pose) — lets you say definitively whether deep learning actually generalizes better than classical methods on *your* data
- [ ] (Stretch) **Siamese/triplet-loss network** — train an embedding space directly for face verification instead of relying purely on a pretrained one

---

## 6. Extra Additions Beyond the Core Rubric

Pick 3–5, not all — depth beats breadth:

- [ ] **Liveness/anti-spoofing check** — reject a printed photo or phone screen (simple version: blink detection over consecutive frames, or texture-based spoof detection)
- [ ] **Real alerting layer** — email (with credentials from an environment variable, never hardcoded/plaintext) **plus** a Telegram bot or Twilio SMS alert, with the snapshot attached and a timestamp
- [ ] **Incident logging** — every "unknown" detection logged to SQLite/MySQL (timestamp, snapshot path, confidence score) — a queryable audit trail instead of just live alerts
- [ ] **Dashboard** — lightweight Streamlit or Flask page showing the live feed, a table of past intrusion events, and a simple analytics chart
- [ ] **Security hardening** if any DB/login layer is kept — parameterized SQL queries (never string-formatted), hashed passwords (`bcrypt`), secrets from environment variables/`.env`
- [ ] **Edge/deployment angle** — export the best model to ONNX/TFLite, note CPU inference time — adds a systems-thinking dimension
- [ ] **Ethics/privacy paragraph** — consent for collecting teammates' faces, data retention/deletion policy, bias limitations of a small homogeneous training set

---

## 7. Suggested Team Split (Review 1 → Review 2)

Example mapping for a 5-person team — add or remove rows to match your actual headcount (pull extra Review 1 extractors from the extended list in Section 4, and split/merge Review 2 roles as needed):

| Person | Review 1 | Review 2 |
|---|---|---|
| 1 | Eigenfaces | CNN-from-scratch |
| 2 | Fisherfaces | Transfer learning / embeddings |
| 3 | LBP | Open-set thresholding + FAR/FRR evaluation |
| 4 | HOG | Grad-CAM / explainability + robustness testing |
| 5 | Gabor | Alerting / dashboard / deployment layer (final live demo) |
| 6+ | Pick from extended options (SIFT/ORB, Haar-like, DCT/Wavelet, color histogram, LPQ, Zernike moments) | Siamese/triplet-loss network, edge deployment, additional extra features from Section 6 |

Build the shared preprocessing pipeline and evaluation harness together before splitting off — that's what makes every member's results directly comparable regardless of team size.

---

## 8. Report / Presentation Structure (both reviews)

1. Problem framing (open-set, not closed-set)
2. Dataset description & collection methodology
3. Preprocessing pipeline
4. Per-feature/method results (table + confusion matrix + ROC)
5. Cross-method comparison
6. Robustness / generalization experiments
7. *(Review 2 adds)* Deep learning results, explainability, final classical-vs-deep verdict
8. Limitations & ethics
9. Live demo

---

## 9. End-to-End Checklist (Scratch → Finish)

- [ ] Repo + folder structure + shared `requirements.txt` set up
- [ ] Data contract for feature extractors agreed
- [ ] Self-collected dataset (one known class per enrolled member + unknown pool) captured
- [ ] AT&T/LFW supplemental dataset downloaded
- [ ] Train/val/test + held-out unknown splits created
- [ ] Augmentation pipeline written
- [ ] Shared face detection + alignment + normalization pipeline built and applied to all data
- [ ] One feature extractor implemented per team member (PCA, LDA, LBP, HOG, Gabor, or from the extended options list)
- [ ] Shared classifier/evaluation harness (SVM/KNN/RF, CV, GridSearch, metrics, ROC) built
- [ ] Open-set thresholding implemented and tuned
- [ ] Feature fusion + t-SNE visualization + robustness study + timing comparison done
- [ ] Review 1 report + presentation written
- [ ] CNN-from-scratch trained
- [ ] Transfer-learning/embedding pipeline built with distance-based open-set recognition
- [ ] FAR/FRR + ROC/AUC evaluated
- [ ] Grad-CAM explainability added
- [ ] Unified classical-vs-deep comparison table produced
- [ ] 3–5 extra features chosen and implemented (liveness check, alerting, logging, dashboard, security hardening, edge export)
- [ ] Ethics/privacy section written
- [ ] Review 2 report + presentation + live demo prepared
