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
  /src/features/{bsif,lpq,wld,gabor,facemesh}.py # one file per person, classifier trained inside each
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
2. [ ] **Alignment** — detect eye landmarks (dlib 68-point or MTCNN 5-point), rotate so eyes are horizontal. This measurably boosts the texture/phase-based descriptors (BSIF, LPQ, WLD, Gabor) since they're sensitive to misalignment; MediaPipe Face Mesh derives its own landmark geometry per face, so it's comparatively more tolerant.
3. [ ] **Normalization** — crop to bounding box + margin, resize to a fixed size (e.g., 128×128), convert to grayscale for classical features (keep color for deep learning), apply CLAHE histogram equalization to reduce lighting variance.
4. [ ] Save processed faces to `/data/processed/` so every feature-extractor script reads from the **same clean input** — required for a fair comparison across methods.

---

## 4. Review 1 — Classical Feature Extraction + Classifiers

Assign **one feature+classifier pipeline per team member**, using the table below — each descriptor is deliberately paired with a classifier suited to its feature type (tree ensembles for the high-dimensional texture histograms, RBF-SVM for the Gabor bank, an MLP for the compact geometric vector), rather than testing every feature against one shared classifier. Each write-up should include: **theory summary, implementation, feature vector dimensionality, chosen classifier's hyperparameters, and results.**

If your team is bigger than 5, extend the table using the same philosophy — pick another texture/phase/geometric descriptor (e.g. Local Ternary Patterns, POEM, SURF/ORB + Bag-of-Visual-Words, HOG, or classic PCA/LDA as a baseline) paired with a classifier not already used (e.g. CatBoost, Extra Trees, Naive Bayes, k-NN). If smaller, merge two rows onto one member, or drop the MediaPipe row last since it captures the most complementary (shape, not texture) information of the five.

### Features Extraction

| Member | Feature Extractor                               | **Actual Features Extracted (Examples)**                                                                                                                                                         | Why these features matter                                                                                                                                                                                        | Classifier    |
| ------ | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **1**  | **BSIF (Binarized Statistical Image Features)** | • Forehead skin texture<br>• Cheek texture<br>• Beard/stubble pattern<br>• Eyebrow texture<br>• Wrinkle patterns around eyes<br>• Lip texture                                                    | Learns statistically meaningful texture patterns from image patches using learned filters instead of handcrafted rules. These micro-texture patterns are distinctive between people. ([PubMed Central (PMC)][1]) | Random Forest |
| **2**  | **LPQ (Local Phase Quantization)**              | • Phase pattern around eye corners<br>• Phase pattern of nostrils<br>• Phase transitions around lips<br>• Eyebrow phase structure<br>• Local phase around chin                                   | Encodes the **local Fourier phase** rather than intensity, making it robust to blur while preserving structural information in facial regions. ([PubMed Central (PMC)][2])                                       | XGBoost       |
| **3**  | **Weber Local Descriptor (WLD)**                | • Relative brightness around eyes<br>• Contrast around nose bridge<br>• Lip-to-skin intensity change<br>• Eyebrow-to-forehead contrast<br>• Shadow transitions on cheeks                         | Measures **relative intensity changes** (contrast perceived by the human eye) rather than absolute brightness, making it more robust to illumination changes. ([IIETA][3])                                       | LightGBM      |
| **4**  | **Gabor Wavelet Bank**                          | • Vertical wrinkles<br>• Horizontal forehead lines<br>• Diagonal eye-edge textures<br>• Nose ridge orientation<br>• Lip edge orientations<br>• Facial hair orientation                           | Extracts **orientation-specific frequency responses** at multiple scales, capturing line and ridge patterns in different directions, similar to processing in the human visual cortex. ([arXiv][4])              | RBF-SVM       |
| **5**  | **MediaPipe Face Mesh (Geometric Features)**    | • Distance between eyes<br>• Nose width<br>• Nose length<br>• Mouth width<br>• Jaw width<br>• Face height<br>• Chin angle<br>• Eye aspect ratio<br>• Mouth aspect ratio<br>• Face symmetry score | Extracts the **shape and proportions** of the face rather than its appearance. These geometric measurements remain informative even when textures change. ([arXiv][4])                                           | MLP           |

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7865363/?utm_source=chatgpt.com "Multi-Block Color-Binarized Statistical Images for Single-Sample Face Recognition - PMC"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10011767/?utm_source=chatgpt.com "Statistical local descriptors for face recognition: a comprehensive study - PMC"
[3]: https://www.iieta.org/journals/ria/paper/10.18280/ria.340501?utm_source=chatgpt.com "Novel Descriptors for Effective Recognition of Face and Facial Expressions | IIETA"
[4]: https://arxiv.org/abs/0907.4984?utm_source=chatgpt.com "Automatic local Gabor Features extraction for face recognition"

> **Implementation note:** BSIF, LPQ, and WLD don't have single mainstream pip packages the way PCA/HOG do — budget time to port a reference implementation from the cited papers, or use a smaller community package. `mediapipe` (Face Mesh), `scikit-learn`, `xgboost`, and `lightgbm` are all mainstream and just need adding to `requirements.txt`.

### For every feature+classifier pipeline, each member should:
- [ ] Run **k-fold cross-validation** — report mean ± std accuracy, not a single number
- [ ] Report **precision, recall, F1, confusion matrix** per class — accuracy alone hides intruder-detection failures (e.g. 95% accuracy while every "unknown" is still misclassified as a known person)
- [ ] Tune their own classifier's hyperparameters with `GridSearchCV`/`RandomizedSearchCV` (Random Forest: `n_estimators`/`max_depth`; XGBoost/LightGBM: `learning_rate`/`num_leaves`/`max_depth`; RBF-SVM: `C`/`gamma`; MLP: hidden layer sizes/activation/learning rate)
- [ ] Implement **open-set thresholding** on top of their classifier's predicted probability/decision score — if confidence falls below a threshold, output "Unknown" instead of forcing a class. Tune the threshold on validation data and report an **ROC curve** for known-vs-unknown separation

### Extra additions that raise the ceiling of Review 1
- [ ] **Dimensionality reduction before classification** — BSIF/LPQ/WLD/Gabor histograms can run into thousands of bins; apply PCA as a second stage and plot accuracy vs. #components
- [ ] **Feature fusion** — concatenate 2–3 of the five descriptors (e.g. BSIF + WLD, or all four texture descriptors + the MediaPipe geometric vector) and retrain one classifier on the combined vector; show whether fusion beats any single pipeline. Ties the whole team's individual work into one shared experiment
- [ ] **t-SNE/PCA 2D visualization** of each feature space colored by identity (`sklearn.manifold.TSNE`) — visually strong for the report/presentation, cheap to produce
- [ ] **Robustness mini-study** — test each feature+classifier pipeline's accuracy under synthetic occlusion (black rectangle over eyes/mouth) or lighting shift
- [ ] **Timing/efficiency comparison** — extraction time + classifier inference time per pipeline, foreshadowing the real-time/deployment discussion in Review 2

---

## 5. Review 2 — Deep Learning

### 5.1 Two parallel tracks (deliberately mirrors Review 1's structure)
1. [ ] **CNN from scratch** — small architecture (3–4 conv blocks + FC), trained directly on your face crops. Will likely *underperform* transfer learning given your small dataset — that gap itself is a good discussion point.
2. [ ] **Transfer learning / pretrained embeddings** — use a pretrained face embedding network (FaceNet, ArcFace via `deepface` or `insightface`, or a pretrained ResNet/MobileNet backbone) to get a fixed-length embedding per face, then — following the same feature-to-classifier pairing philosophy as Review 1 — feed it into a classifier suited to the embedding (e.g. RBF-SVM or an MLP) OR fine-tune the last few layers end-to-end.

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

Example mapping for a 5-person team — add or remove rows to match your actual headcount (pull extra Review 1 descriptor+classifier pairs using the scaling guidance in Section 4, and split/merge Review 2 roles as needed):

| Person | Review 1 (feature + classifier) | Review 2 |
|---|---|---|
| 1 | BSIF + Random Forest | CNN-from-scratch |
| 2 | LPQ + XGBoost | Transfer learning / embeddings |
| 3 | Weber Local Descriptor (WLD) + LightGBM | Open-set thresholding + FAR/FRR evaluation |
| 4 | Gabor Wavelet Bank + RBF-SVM | Grad-CAM / explainability + robustness testing |
| 5 | MediaPipe Face Mesh (geometric) + MLP | Alerting / dashboard / deployment layer (final live demo) |
| 6+ | Another descriptor+classifier pair (e.g. Local Ternary Patterns, POEM, SURF/ORB+BoVW, HOG, or classic PCA/LDA, paired with CatBoost/Extra Trees/Naive Bayes/k-NN) | Siamese/triplet-loss network, edge deployment, additional extra features from Section 6 |

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
- [ ] One feature+classifier pipeline implemented per team member (BSIF+Random Forest, LPQ+XGBoost, WLD+LightGBM, Gabor Wavelet Bank+RBF-SVM, MediaPipe Face Mesh+MLP, or additional pairs if the team is larger than 5)
- [ ] Shared evaluation harness built (each member's own classifier — RF/XGBoost/LightGBM/RBF-SVM/MLP — plugged in via CV, hyperparameter search, metrics, ROC)
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
