# Benchmark Results & Metrics Summary

### 5 Independent Classical CV & ML Pipeline Comparison

| Model / Pipeline                       | Feature Extractor   | Classifier          |   Feature Dim | Train Acc (%)   | Val Acc (%)   | Test Acc (%)   |   Precision |   Recall |   F1-Score | Latency (ms/face)   |
|:---------------------------------------|:--------------------|:--------------------|--------------:|:----------------|:--------------|:---------------|------------:|---------:|-----------:|:--------------------|
| Pipeline 1 (BSIF + Random Forest)      | BSIF                | Random Forest       |          4096 | 100.0%          | 59.7%         | 73.6%          |      0.8286 |   0.7222 |     0.7326 | 6.60 ms             |
| Pipeline 2 (LPQ + k-NN)                | LPQ                 | k-NN                |          4096 | 100.0%          | 83.3%         | 75.0%          |      0.7872 |   0.7278 |     0.7259 | 12.14 ms            |
| Pipeline 3 (WLD + Logistic Regression) | WLD                 | Logistic Regression |          2048 | 100.0%          | 80.6%         | 76.4%          |      0.8573 |   0.7444 |     0.7656 | 4.57 ms             |
| Pipeline 4 (Gabor + RBF-SVM)           | Gabor Wavelets      | RBF-SVM             |          1920 | 100.0%          | 76.4%         | 76.4%          |      0.7796 |   0.7444 |     0.7215 | 38.72 ms            |
| Pipeline 5 (Geometry + Decision Tree)  | Landmark Geometry   | Decision Tree       |           745 | 100.0%          | 76.4%         | 76.4%          |      0.8278 |   0.7444 |     0.7499 | 2.90 ms             |

### Open-Set Intruder Recognition Performance

- **Correct Identification Rate (CIR)**: 9.72%
- **False Rejection Rate (FRR)**: 90.28%
- **True Intruder Detection Rate**: 100.00%
- **False Acceptance Rate (FAR)**: 0.00%


### Confusion Matrix Artifacts

- Test Confusion Matrices: `reports/figures/confusion_matrices_test.png`
- Train vs Test Comparative Confusion Matrices: `reports/figures/confusion_matrices_train_vs_test.png`
