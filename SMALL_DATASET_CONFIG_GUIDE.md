# Hướng dẫn Config cho Dataset Nhỏ (~8,000 tokens)

## 🎯 Tổng quan

Với dataset kích thước **~8,000 tokens** (training ~7,200, validation ~800), cần tối ưu hóa để:
1. **Tránh overfitting** - Mô hình học quá kỹ training data
2. **Tăng generalization** - Mô hình hoạt động tốt trên dữ liệu mới
3. **Tối ưu tốc độ** - Training nhanh với dataset nhỏ

---

## 📊 So sánh Config: Mặc định vs Tối ưu

### **Neural Network (MLP)**

| Tham số | Mặc định (Large Data) | Tối ưu (8K Data) | Lý do |
|---------|----------------------|------------------|-------|
| `NN_EPOCHS` | 50 | **100** | Dataset nhỏ, có thể train lâu hơn |
| `NN_BATCH_SIZE` | 128 | **64** | Batch nhỏ → nhiều updates → học tốt hơn |
| `NN_HIDDEN_DIMS` | 512,256,128 | **256,128** | Mô hình đơn giản hơn → ít overfit |
| `NN_DROPOUT` | 0.3 | **0.4** | Dropout cao hơn → regularization mạnh |

**Giải thích chi tiết:**
- ✅ **Batch size 64**: Với 7,200 samples, mỗi epoch có ~112 updates (vs 56 updates với batch 128)
- ✅ **Hidden dims 256,128**: Giảm từ ~500K parameters xuống ~200K parameters
- ✅ **Dropout 0.4**: Tắt ngẫu nhiên 40% neurons → buộc mô hình học robust features

---

### **Random Forest**

| Tham số | Mặc định (Large Data) | Tối ưu (8K Data) | Lý do |
|---------|----------------------|------------------|-------|
| `RF_N_ESTIMATORS` | 200 | **150** | Ít trees → nhanh hơn, ít overfit |
| `RF_MAX_DEPTH` | None (unlimited) | **15** | Giới hạn độ sâu → tránh học noise |
| `RF_MIN_SAMPLES_SPLIT` | 5 | **10** | Cần nhiều samples hơn để split |
| `RF_MIN_SAMPLES_LEAF` | 2 | **4** | Leaf phải có ít nhất 4 samples |

**Giải thích chi tiết:**
- ✅ **Max depth 15**: Với 7,200 samples, depth 15 đủ để học patterns mà không overfit
- ✅ **Min samples split 10**: Chỉ split nếu node có ≥10 samples (0.14% của data)
- ✅ **Min samples leaf 4**: Mỗi leaf phải đại diện cho ít nhất 4 samples

---

### **XGBoost** (Khuyến nghị cao nhất)

| Tham số | Mặc định (Large Data) | Tối ưu (8K Data) | Lý do |
|---------|----------------------|------------------|-------|
| `XGB_N_ESTIMATORS` | 200 | **150** | Ít trees với early stopping |
| `XGB_MAX_DEPTH` | 6 | **4** | Trees nông hơn → ít overfit |
| `XGB_LEARNING_RATE` | 0.1 | **0.05** | Học chậm hơn → generalize tốt |
| `XGB_SUBSAMPLE` | 0.8 | **0.7** | Dùng 70% data mỗi tree |
| `XGB_COLSAMPLE_BYTREE` | 0.8 | **0.7** | Dùng 70% features mỗi tree |
| `XGB_GAMMA` | 0 | **0.1** | Yêu cầu loss giảm ≥0.1 để split |
| `XGB_REG_ALPHA` | 0 | **0.1** | L1 regularization |
| `XGB_REG_LAMBDA` | 1 | **2** | L2 regularization mạnh hơn |
| `XGB_EARLY_STOPPING` | 10 | **15** | Kiên nhẫn hơn trước khi dừng |

**Giải thích chi tiết:**
- ✅ **Max depth 4**: Trees nông → mỗi tree học simple patterns → ensemble mạnh
- ✅ **Learning rate 0.05**: Mỗi tree đóng góp ít hơn → cần nhiều trees → robust hơn
- ✅ **Subsample 0.7**: Mỗi tree chỉ thấy 70% data → giống bagging → tránh overfit
- ✅ **Gamma 0.1**: Chỉ split nếu loss giảm đáng kể → tránh splits không cần thiết
- ✅ **Reg_lambda 2**: Penalty mạnh cho weights lớn → smooth predictions

---

## 🎓 Nguyên tắc chung cho Small Dataset

### **1. Giảm Model Complexity**
```
Lý do: Mô hình phức tạp dễ học thuộc lòng training data
Cách làm:
  - Neural Network: Ít layers, ít neurons
  - Random Forest: Giới hạn depth, tăng min_samples
  - XGBoost: Shallow trees, strong regularization
```

### **2. Tăng Regularization**
```
Lý do: Ngăn mô hình fit quá sát với training data
Cách làm:
  - Neural Network: Dropout cao (0.4-0.5)
  - Random Forest: Min_samples_split/leaf cao
  - XGBoost: L1/L2 regularization, gamma
```

### **3. Giảm Batch Size (Neural Network)**
```
Lý do: Nhiều updates hơn mỗi epoch → học tốt hơn
Công thức: batch_size ≈ sqrt(training_size)
  - 7,200 samples → batch ~64-85
```

### **4. Tăng Training Time**
```
Lý do: Dataset nhỏ → mỗi epoch nhanh → có thể train lâu
Cách làm:
  - Neural Network: Tăng epochs (50 → 100)
  - XGBoost: Tăng early_stopping_rounds (10 → 15)
```

---

## 📈 Kỳ vọng Accuracy với Config Tối ưu

### **Baseline (Logistic Regression)**
- Top 1: **62.23%** ✅ (đã chạy)
- Top 5: **92.66%** ✅ (đã chạy)

### **Với Config Tối ưu:**

| Phương pháp | Top 1 (dự kiến) | Top 5 (dự kiến) | Cải thiện |
|-------------|-----------------|-----------------|-----------|
| **Neural Network** | 68-73% | 94-96% | +6-11% |
| **Random Forest** | 70-76% | 95-97% | +8-14% |
| **XGBoost** | 72-78% | 96-98% | +10-16% 🏆 |

---

## ⚠️ Dấu hiệu Overfitting cần chú ý

### **Khi chạy training, nếu thấy:**

1. **Training accuracy >> Validation accuracy**
   ```
   Ví dụ: Train 95%, Validation 65%
   → Overfitting nghiêm trọng!
   ```
   **Giải pháp:**
   - Tăng dropout (NN)
   - Giảm max_depth (RF, XGB)
   - Tăng regularization (XGB)

2. **Validation accuracy giảm sau vài epochs**
   ```
   Epoch 20: Val 70%
   Epoch 30: Val 72%
   Epoch 40: Val 69% ← Bắt đầu overfit
   ```
   **Giải pháp:**
   - Early stopping sẽ tự động dừng
   - Giảm số epochs/estimators

3. **Perfect training accuracy (100%)**
   ```
   → Mô hình học thuộc lòng data!
   ```
   **Giải pháp:**
   - Tăng regularization mạnh hơn
   - Giảm model complexity

---

## 🚀 Cách chạy với Config mới

### **Bước 1: Reload config**
```bash
# Config đã được cập nhật tự động
source config.env
```

### **Bước 2: Chạy từng phương pháp**

**XGBoost (Khuyến nghị cao nhất):**
```bash
bash scripts/train_set/classifier_mapping/xgboost_classifier.sh
```

**Neural Network:**
```bash
bash scripts/train_set/classifier_mapping/neural_network.sh
```

**Random Forest:**
```bash
bash scripts/train_set/classifier_mapping/random_forest.sh
```

### **Bước 3: Xem kết quả**
```bash
# Thay đổi fileDir trong get_prediction_stat.sh cho từng method:
# result_xgb, result_nn, result_rf

bash scripts/train_set/classifier_mapping/get_prediction_stat.sh
```

---

## 💡 Tips Nâng cao

### **1. Grid Search cho XGBoost**
Nếu muốn tìm config tốt nhất:
```bash
# Thử các giá trị khác nhau:
XGB_MAX_DEPTH=3,4,5
XGB_LEARNING_RATE=0.03,0.05,0.1
XGB_REG_LAMBDA=1,2,3
```

### **2. Ensemble nhiều models**
```python
# Kết hợp predictions từ 3 models
final_prediction = vote(xgb_pred, rf_pred, nn_pred)
```

### **3. Data Augmentation**
Nếu có thể, tăng data bằng cách:
- Synonym replacement
- Back-translation
- Paraphrasing

---

## 📊 Monitoring Training

### **Neural Network:**
```
Epoch [10/100], Loss: 2.1234, Accuracy: 65.23%
Epoch [20/100], Loss: 1.8765, Accuracy: 68.45%
...
```
- Loss giảm đều → Tốt ✅
- Loss tăng → Overfitting ⚠️

### **XGBoost:**
```
[0]     validation-mlogloss:2.1234
[10]    validation-mlogloss:1.8765
[20]    validation-mlogloss:1.7654
```
- mlogloss giảm đều → Tốt ✅
- mlogloss tăng → Early stopping sẽ dừng ✅

---

## 🎯 Kết luận

Với dataset **~8,000 tokens**, config đã được tối ưu để:
- ✅ **Tránh overfitting** với regularization mạnh
- ✅ **Tăng generalization** với model đơn giản hơn
- ✅ **Training nhanh** với ít parameters hơn

**Khuyến nghị:** Chạy XGBoost trước, sau đó so sánh với Neural Network và Random Forest.

Chúc bạn đạt accuracy cao! 🚀
