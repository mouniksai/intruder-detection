# Benchmark Results & Metrics Summary

### 25 Classical CV & ML Experiments (5 Features x 5 Classifiers)

| Feature Extractor   | Classifier          |   Feature Dim | Train Acc (%)   | Val Acc (%)   | Test Acc (%)   |   Precision |   Recall |   F1-Score | Latency (ms/face)   |
|:--------------------|:--------------------|--------------:|:----------------|:--------------|:---------------|------------:|---------:|-----------:|:--------------------|
| BSIF                | Random Forest       |          4096 | 100.0%          | 59.7%         | 73.6%          |      0.8286 |   0.7222 |     0.7326 | 7.03 ms             |
| BSIF                | KNN                 |          4096 | 100.0%          | 73.6%         | 66.7%          |      0.78   |   0.6444 |     0.6565 | 0.74 ms             |
| BSIF                | Logistic Regression |          4096 | 100.0%          | 81.9%         | 77.8%          |      0.8483 |   0.7667 |     0.7671 | 0.12 ms             |
| BSIF                | SVM                 |          4096 | 100.0%          | 70.8%         | 72.2%          |      0.7911 |   0.7111 |     0.7067 | 0.53 ms             |
| BSIF                | Decision Tree       |          4096 | 100.0%          | 62.5%         | 55.6%          |      0.5806 |   0.5278 |     0.5163 | 0.17 ms             |
| LPQ                 | Random Forest       |          4096 | 100.0%          | 72.2%         | 63.9%          |      0.6404 |   0.6167 |     0.5991 | 10.84 ms            |
| LPQ                 | KNN                 |          4096 | 100.0%          | 83.3%         | 75.0%          |      0.7872 |   0.7278 |     0.7259 | 0.54 ms             |
| LPQ                 | Logistic Regression |          4096 | 100.0%          | 81.9%         | 81.9%          |      0.8444 |   0.8167 |     0.8044 | 0.01 ms             |
| LPQ                 | SVM                 |          4096 | 100.0%          | 83.3%         | 75.0%          |      0.8078 |   0.7444 |     0.7281 | 0.51 ms             |
| LPQ                 | Decision Tree       |          4096 | 100.0%          | 72.2%         | 68.1%          |      0.6667 |   0.6611 |     0.643  | 0.39 ms             |
| WLD                 | Random Forest       |          2048 | 100.0%          | 70.8%         | 69.4%          |      0.7839 |   0.6667 |     0.6817 | 4.63 ms             |
| WLD                 | KNN                 |          2048 | 100.0%          | 73.6%         | 70.8%          |      0.7587 |   0.6833 |     0.6824 | 0.21 ms             |
| WLD                 | Logistic Regression |          2048 | 100.0%          | 80.6%         | 76.4%          |      0.8573 |   0.7444 |     0.7656 | 0.08 ms             |
| WLD                 | SVM                 |          2048 | 100.0%          | 76.4%         | 72.2%          |      0.7189 |   0.7    |     0.6787 | 0.22 ms             |
| WLD                 | Decision Tree       |          2048 | 100.0%          | 79.2%         | 70.8%          |      0.7444 |   0.6889 |     0.6826 | 0.37 ms             |
| Gabor               | Random Forest       |          1920 | 100.0%          | 73.6%         | 75.0%          |      0.7611 |   0.7278 |     0.726  | 37.45 ms            |
| Gabor               | KNN                 |          1920 | 100.0%          | 70.8%         | 75.0%          |      0.7883 |   0.7333 |     0.721  | 0.66 ms             |
| Gabor               | Logistic Regression |          1920 | 100.0%          | 81.9%         | 76.4%          |      0.7911 |   0.7444 |     0.7302 | 0.44 ms             |
| Gabor               | SVM                 |          1920 | 100.0%          | 76.4%         | 76.4%          |      0.7796 |   0.7444 |     0.7215 | 0.19 ms             |
| Gabor               | Decision Tree       |          1920 | 100.0%          | 80.6%         | 75.0%          |      0.7806 |   0.7278 |     0.7175 | 0.34 ms             |
| Geometry            | Random Forest       |           745 | 100.0%          | 73.6%         | 69.4%          |      0.6836 |   0.6833 |     0.6549 | 2.88 ms             |
| Geometry            | KNN                 |           745 | 100.0%          | 76.4%         | 83.3%          |      0.8744 |   0.8167 |     0.8102 | 0.48 ms             |
| Geometry            | Logistic Regression |           745 | 100.0%          | 86.1%         | 81.9%          |      0.8417 |   0.8    |     0.7921 | 0.35 ms             |
| Geometry            | SVM                 |           745 | 100.0%          | 80.6%         | 76.4%          |      0.8144 |   0.7556 |     0.7428 | 0.10 ms             |
| Geometry            | Decision Tree       |           745 | 100.0%          | 76.4%         | 76.4%          |      0.8278 |   0.7444 |     0.7499 | 0.17 ms             |

### Open-Set Intruder Recognition Performance

- **Correct Identification Rate (CIR)**: 9.72%
- **False Rejection Rate (FRR)**: 90.28%
- **True Intruder Detection Rate**: 100.00%
- **False Acceptance Rate (FAR)**: 0.00%


### Confusion Matrix & Heatmap Artifacts

- 25 Experiments Test Confusion Matrix Grid: `reports/figures/confusion_matrices_test.png`
- 5x5 Accuracy Matrix Heatmap: `reports/figures/benchmark_25_matrix.png`
- Train vs Test Comparative Confusion Matrices: `reports/figures/confusion_matrices_train_vs_test.png`
