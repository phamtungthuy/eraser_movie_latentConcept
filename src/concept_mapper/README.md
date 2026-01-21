# Advanced Concept Mappers

Bộ công cụ này cung cấp các phương pháp nâng cao để ánh xạ từ representation của từ (word embeddings) sang các concept tương ứng.

## Tổng quan các phương pháp

### 1. **Logistic Regression** (Baseline)
- **File**: `logistic_regression.py`
- **Script**: `logistic_regression.sh`
- **Mô tả**: Phương pháp cơ bản sử dụng hồi quy logistic tuyến tính
- **Ưu điểm**: 
  - Nhanh, đơn giản
  - Dễ giải thích
  - Ít tham số
- **Nhược điểm**: 
  - Chỉ học được quan hệ tuyến tính
  - Hiệu suất hạn chế với dữ liệu phức tạp

### 2. **Neural Network (MLP)** ⭐ Mới
- **File**: `neural_network.py`
- **Script**: `neural_network.sh`
- **Mô tả**: Mạng neural đa lớp với PyTorch
- **Kiến trúc**:
  - Nhiều hidden layers có thể cấu hình
  - Batch Normalization để ổn định training
  - Dropout để tránh overfitting
  - ReLU activation functions
- **Ưu điểm**:
  - Học được các quan hệ phi tuyến phức tạp
  - Linh hoạt trong kiến trúc
  - Tận dụng GPU để training nhanh
- **Nhược điểm**:
  - Cần nhiều dữ liệu hơn
  - Khó giải thích
  - Cần tune hyperparameters

**Tham số cấu hình** (trong `config.env`):
```bash
NN_EPOCHS=50                    # Số epoch training
NN_BATCH_SIZE=128              # Kích thước batch
NN_LEARNING_RATE=0.001         # Learning rate
NN_HIDDEN_DIMS=512,256,128     # Kích thước các hidden layers
NN_DROPOUT=0.3                 # Tỷ lệ dropout
NN_DEVICE=cuda                 # Device (cuda hoặc cpu)
```

### 3. **Random Forest** ⭐ Mới
- **File**: `random_forest.py`
- **Script**: `random_forest.sh`
- **Mô tả**: Ensemble learning với nhiều decision trees
- **Đặc điểm**:
  - Kết hợp nhiều decision trees
  - Voting để đưa ra dự đoán cuối cùng
  - Tính toán feature importance
- **Ưu điểm**:
  - Robust, ít bị overfitting
  - Không cần chuẩn hóa dữ liệu
  - Xử lý tốt với high-dimensional data
  - Cung cấp feature importance
- **Nhược điểm**:
  - Tốn bộ nhớ với nhiều trees
  - Chậm hơn trong inference
  - Khó giải thích từng prediction

**Tham số cấu hình** (trong `config.env`):
```bash
RF_N_ESTIMATORS=200            # Số lượng trees
RF_MAX_DEPTH=None              # Độ sâu tối đa của tree (None = không giới hạn)
RF_MIN_SAMPLES_SPLIT=5         # Số samples tối thiểu để split
RF_MIN_SAMPLES_LEAF=2          # Số samples tối thiểu tại leaf
RF_MAX_FEATURES=sqrt           # Số features xem xét khi split
RF_N_JOBS=-1                   # Số CPU cores (-1 = tất cả)
```

### 4. **XGBoost** ⭐ Mới (Khuyến nghị)
- **File**: `xgboost_classifier.py`
- **Script**: `xgboost_classifier.sh`
- **Mô tả**: Gradient boosting hiện đại, state-of-the-art
- **Đặc điểm**:
  - Gradient boosting với regularization
  - Early stopping tự động
  - Histogram-based algorithm cho tốc độ
  - L1/L2 regularization
- **Ưu điểm**:
  - Hiệu suất cao nhất trong hầu hết trường hợp
  - Xử lý tốt missing values
  - Built-in regularization
  - Tối ưu về tốc độ và bộ nhớ
  - Feature importance
- **Nhược điểm**:
  - Nhiều hyperparameters cần tune
  - Có thể overfit nếu không cẩn thận

**Tham số cấu hình** (trong `config.env`):
```bash
XGB_N_ESTIMATORS=200           # Số boosting rounds
XGB_MAX_DEPTH=6                # Độ sâu tối đa của tree
XGB_LEARNING_RATE=0.1          # Learning rate (eta)
XGB_SUBSAMPLE=0.8              # Tỷ lệ subsample của training data
XGB_COLSAMPLE_BYTREE=0.8       # Tỷ lệ subsample của features
XGB_GAMMA=0                    # Minimum loss reduction
XGB_REG_ALPHA=0                # L1 regularization
XGB_REG_LAMBDA=1               # L2 regularization
XGB_N_JOBS=-1                  # Số CPU cores
XGB_EARLY_STOPPING=10          # Early stopping rounds
```

## Cách sử dụng

### 1. Cấu hình tham số
Thêm các tham số vào file `config.env`:

```bash
# Model configuration
MODEL=google-bert/bert-base-cased
LAYER=12

# Neural Network parameters (optional)
NN_EPOCHS=50
NN_BATCH_SIZE=128
NN_LEARNING_RATE=0.001
NN_HIDDEN_DIMS=512,256,128
NN_DROPOUT=0.3
NN_DEVICE=cuda

# Random Forest parameters (optional)
RF_N_ESTIMATORS=200
RF_MAX_DEPTH=None
RF_MIN_SAMPLES_SPLIT=5
RF_MIN_SAMPLES_LEAF=2
RF_MAX_FEATURES=sqrt
RF_N_JOBS=-1

# XGBoost parameters (optional)
XGB_N_ESTIMATORS=200
XGB_MAX_DEPTH=6
XGB_LEARNING_RATE=0.1
XGB_SUBSAMPLE=0.8
XGB_COLSAMPLE_BYTREE=0.8
XGB_GAMMA=0
XGB_REG_ALPHA=0
XGB_REG_LAMBDA=1
XGB_N_JOBS=-1
XGB_EARLY_STOPPING=10
```

### 2. Chạy training

#### Neural Network:
```bash
bash scripts/train_set/classifier_mapping/neural_network.sh
```

#### Random Forest:
```bash
bash scripts/train_set/classifier_mapping/random_forest.sh
```

#### XGBoost:
```bash
bash scripts/train_set/classifier_mapping/xgboost_classifier.sh
```

### 3. Kết quả

Mỗi phương pháp sẽ tạo ra:
- **Model file**: `eraser_movie/{model_name}/result_{method}/model/layer_{layer}_{method}_classifier.pkl`
- **Validation predictions**: `eraser_movie/{model_name}/result_{method}/validate_predictions/predictions_layer_{layer}.csv`
- **Accuracy score**: In ra console

## So sánh hiệu suất

| Phương pháp | Tốc độ Training | Tốc độ Inference | Accuracy (dự kiến) | Bộ nhớ | Khả năng giải thích |
|-------------|----------------|------------------|-------------------|--------|-------------------|
| Logistic Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Neural Network | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Random Forest | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Khuyến nghị

1. **Bắt đầu với XGBoost**: Thường cho kết quả tốt nhất với ít effort
2. **Thử Neural Network**: Nếu có GPU và muốn tận dụng deep learning
3. **Random Forest**: Nếu cần feature importance và robust model
4. **Logistic Regression**: Baseline để so sánh

## Dependencies

Cài đặt các thư viện cần thiết:

```bash
pip install torch torchvision  # Cho Neural Network
pip install scikit-learn       # Cho Random Forest
pip install xgboost           # Cho XGBoost
pip install pandas numpy dill
```

## Troubleshooting

### Neural Network không train được
- Kiểm tra CUDA có sẵn: `python -c "import torch; print(torch.cuda.is_available())"`
- Nếu không có GPU, đổi `NN_DEVICE=cpu` trong config.env
- Giảm batch size nếu bị out of memory

### Random Forest/XGBoost chậm
- Giảm `n_estimators`
- Tăng `min_samples_split` (RF) hoặc `max_depth` (XGB)
- Đảm bảo `n_jobs=-1` để dùng tất cả CPU cores

### Out of Memory
- Giảm batch size (NN)
- Giảm số trees (RF, XGB)
- Giảm max_depth (RF, XGB)
