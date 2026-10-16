# BÁO CÁO ĐÁNH GIÁ CHI TIẾT 11 CHỈ SỐ: MÔ HÌNH 50 EPOCHS

> **Tệp trọng số**: `best_model_50epochs.pth`  
> **Tập kiểm thử**: 224 ảnh (`SD/data/Testing`: 112 Smoking, 112 NotSmoking)  
> **Ngưỡng quyết định**: $\tau = 0.5$  

---

## 1. Bảng đối chiếu 11 chỉ số khoa học so với bài báo gốc (Table 3)

| Chỉ số (Metric) | Công thức (Section 6.2) | Kết quả 50 Epochs | Bài báo công bố (Inception-ResNet-V2) | Chênh lệch |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives ($T_p$)** | $T_p$ | **105** | 109 | -4 |
| **True Negatives ($T_n$)** | $T_n$ | **96** | 108 | -12 |
| **False Positives ($F_p$)** | $F_p$ | **16** | 4 | +12 |
| **False Negatives ($F_n$)** | $F_n$ | **7** | 3 | +4 |
| **Độ chính xác (Accuracy)** | $\frac{T_p + T_n}{\text{Total}}$ (Eq. 4) | **89.73%** | 96.87% | -7.14% |
| **Độ chuẩn xác (Precision - $PV_p$)** | $\frac{T_p}{T_p + F_p}$ (Eq. 5) | **86.78%** | 97.32% | -10.54% |
| **Độ nhạy (Recall / Sensitivity)** | $\frac{T_p}{T_p + F_n}$ (Eq. 6) | **93.75%** | 96.46% | -2.71% |
| **Độ đặc hiệu (Specificity)** | $\frac{T_n}{T_n + F_p}$ (Eq. 7) | **85.71%** | 96.43% | -10.72% |
| **Negative Pred Value ($PV_n$)** | $\frac{T_n}{T_n + F_n}$ (Eq. 8) | **93.20%** | 97.29% | -4.09% |
| **Tỷ lệ báo động giả ($FPR$)** | $1 - \text{Specificity}$ (Eq. 9) | **14.29%** | 3.57% | +10.72% |
| **Tỷ lệ bỏ sót ($FNR$)** | $1 - \text{Sensitivity}$ (Eq. 10) | **6.25%** | 3.54% | +2.71% |
| **Tỷ lệ phát hiện sai ($FDR$)** | $1 - PV_p$ (Eq. 11) | **13.22%** | 2.68% | +10.54% |
| **Điểm $F_1$ ($F_1\text{-Score}$)** | $2 \times \frac{P \times R}{P + R}$ (Eq. 12) | **0.9013** | 0.9688 | -0.0675 |
| **Tỷ lệ lỗi ($E_r$)** | $\frac{F_p + F_n}{\text{Total}}$ (Eq. 13) | **10.27%** | 3.12% | +7.15% |
| **Diện tích dưới ROC ($AUC$)** | Area under ROC | **0.9621** | 0.9859 | -0.0238 |
| **Độ chính xác TB ($AP$)** | Area under PR | **0.9632** | 0.9848 | -0.0216 |

---

## 2. Các biểu đồ trực quan hóa khoa học

- **Confusion Matrix**: `confusion_matrix.png`  
- **Đường cong ROC (Full & Zoomed)**: `roc_curve.png` (AUC = 0.9621)  
- **Đường cong Precision-Recall (Full & Scaled)**: `precision_recall_curve.png` (AP = 0.9632)  

---

## 3. Danh sách các mẫu phân loại sai (Error Analysis)

### A. False Positives (16 ảnh - Thực tế: NotSmoking, Đoán nhầm: Smoking)
| STT | Tên tệp ảnh | Xác suất Smoking |
| :---: | :--- | :---: |
| 1 | `notsmoking_0268.jpg` | 0.8806 |
| 2 | `notsmoking_0259.jpg` | 0.7473 |
| 3 | `notsmoking_0032.jpg` | 0.7306 |
| 4 | `notsmoking_0281.jpg` | 0.7212 |
| 5 | `notsmoking_0532.jpg` | 0.7117 |
| 6 | `notsmoking_0130.jpg` | 0.6716 |
| 7 | `notsmoking_0014.jpg` | 0.6194 |
| 8 | `notsmoking_0465.jpg` | 0.6161 |
| 9 | `notsmoking_0041.jpg` | 0.6026 |
| 10 | `notsmoking_0036.jpg` | 0.5937 |
| 11 | `notsmoking_0533.jpg` | 0.5909 |
| 12 | `notsmoking_0388.jpg` | 0.5873 |
| 13 | `notsmoking_0549.jpg` | 0.5735 |
| 14 | `notsmoking_0180.jpg` | 0.5689 |
| 15 | `notsmoking_0107.jpg` | 0.5285 |
| 16 | `notsmoking_0233.jpg` | 0.5279 |

### B. False Negatives (7 ảnh - Thực tế: Smoking, Bỏ sót thành: NotSmoking)
| STT | Tên tệp ảnh | Xác suất Smoking |
| :---: | :--- | :---: |
| 1 | `smoking_0063.jpg` | 0.2828 |
| 2 | `smoking_0546.jpg` | 0.2884 |
| 3 | `smoking_0423.jpg` | 0.3021 |
| 4 | `smoking_0352.jpg` | 0.3519 |
| 5 | `smoking_0484.jpg` | 0.3892 |
| 6 | `smoking_0403.jpg` | 0.4789 |
| 7 | `smoking_0559.jpg` | 0.4800 |
