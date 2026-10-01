# Trong so Mo hinh Tien huan luyen: best_uav_model.keras

Thu muc nay chua hoac theo doi tep trong so toi uu nhat cua mo hinh **Anti-UAV Anchor-Free CenterNet (Chi su dung kenh Visible RGB)**.

## Thong so Ky thuat Mo hinh
* **Kien truc**: ResNet-50v2 Backbone + Coordinate Attention + P2-FPN + Decoupled CenterNet Heads
* **Kich thuoc dau vao**: `(640, 640, 9)` (Bo ba khung hinh Visible RGB lien tiep: $t-1, t, t+1$)
* **Ham kich hoat nhanh du doan Size**: Sigmoid ket hop Logit Prior Initialization `Constant(-2.66)`
* **Dung luong tep**: ~102.8 MB
* **Dinh dang**: Keras SavedModel 3.x (`.keras`)

## Huong dan Tai trong so
Neu tep `best_uav_model.keras` chua co san trong thu muc cuc bo (do gioi han kich thuoc tep 100MB tren GitHub), ban co the su dung mot trong cac cach sau:

### Cach 1: Tai truc tiep tu Kaggle Dataset
Trong so duoc luu tru cong khai tai:
* **[Kaggle Dataset: uav-checkpoint](https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint)**

Sau khi tai ve, dat tep `best_uav_model.keras` truc tiep vao thu muc `weights/` nay.

### Cach 2: Su dung Script Python tu dong
Thuc thi kịch ban tai tu dong:
```bash
python weights/download_weights.py
```

### Cach 3: Su dung Kaggle CLI
```bash
kaggle datasets download -d namnguyen171006/uav-checkpoint -p weights/ --unzip
```
