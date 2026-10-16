# Smoker Detection System (Khan et al., Sensors 2022)

Mã nguồn PyTorch tái hiện bài báo khoa học:  
> **"CNN-Based Smoker Classification and Detection in Smart City Application"**  
> *S. Khan et al., Sensors 2022, 22, 892. DOI: 10.3390/s22030892*

---

## 1. Cấu trúc thư mục (Directory Structure)

```text
SD/
├── data/                      # Bộ dữ liệu ảnh (716 Train, 180 Val, 224 Test)
│   ├── Training/              # [Smoking / NotSmoking]
│   ├── Validation/            # [Smoking / NotSmoking]
│   └── Testing/               # [Smoking / NotSmoking]
├── scrip/                     # Toàn bộ mã nguồn thực thi
│   ├── config.py              # Tham số cấu hình tập trung (Hyperparameters, Paths)
│   ├── dataset.py             # DataLoader & Data Augmentation (Keras-style Nearest Fill)
│   ├── model.py               # Inception-ResNet-V2 (Frozen Backbone + 3-Layer Dense Head)
│   ├── train.py               # Huấn luyện mô hình (Hỗ trợ resume & steps_per_epoch)
│   └── evaluate.py            # Đánh giá 11 chỉ số kiểm thử, vẽ Confusion Matrix, ROC, PR
├── model/                     # Nơi lưu trọng số mô hình (*.pth - bị gitignore vì > 100MB)
├── outputs/                   # Biểu đồ học (Loss/Acc curve) và nhật ký JSON
└── eval/                      # Báo cáo đánh giá chi tiết theo từng mốc huấn luyện
```

---

## 2. Cài đặt môi trường (Installation)

Yêu cầu Python $\ge$ 3.10 và GPU (khuyến nghị CUDA):

```bash
# Cài đặt các thư viện phụ thuộc chính
pip install torch torchvision timm opencv-python scikit-learn matplotlib tqdm
```

---

## 3. Hướng dẫn sử dụng nhanh (Quick Start)

Mọi lệnh đều chạy từ thư mục `SD/scrip`:
```bash
cd SD/scrip
```

### A. Huấn luyện (Training) (Mặc định dùng 2)

* **Chế độ 1: dùng steps-per-epoch**  
  *(50 epochs, mỗi epoch cập nhật 100 batch augment = 5,000 steps gradient)*:
  ```bash
  python train.py --epochs 50 --steps-per-epoch 100
  ```

* **Chế độ 2: Huấn luyện nhanh theo kích thước dataset** *(12 batch / epoch)*:
  ```bash
  python train.py --epochs 50
  ```

* **Chế độ 3: Huấn luyện tiếp tục từ Checkpoint (Resume)**:
  ```bash
  python train.py --epochs 100 --resume ../model/latest_checkpoint.pth
  ```

### B. Đánh giá mô hình trên tập Test (Evaluation)
```bash
python evaluate.py --model-path ../model/best_model.pth --output-dir ../eval/results
```
