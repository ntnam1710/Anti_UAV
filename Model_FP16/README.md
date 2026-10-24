# Huong Dan Su Dung Cac Mo Hinh Luong Tu Hoa va Toi Uu Bien (Model_FP16)

Thu muc nay chua ma nguon thuc nghiem, kich ban chuyen doi va cac mo hinh da duoc toi uu hoa luong tu hoa ban chinh xac FP16 phuc vu trien khai tren thiet bi bien nhung nhu NVIDIA Jetson Nano / Orin Nano.

## 1. Danh Sach Cac Mo Hinh va Dinh Dang

1. **best_uav_model_fp16.tflite (51.08 MB):**
   * Dinh dang: TensorFlow Lite FlatBuffer (FP16).
   * Dac tinh: Giam 50.36% dung luong so voi mo hinh goc FP32 (tu 102.9 MB xuong 51.08 MB).
   * Hieu nang: Toc do suy luan dat 33.8 FPS (nhanh hon 35.74% so voi FP32), vuot nguong chuan thoi gian thuc 30 FPS.
   * Do chinh xac: Bao toan tuyet doi 92.26% mAP@0.5 va 86.42% NWD-mAP cho drone sieu nho.
   * Su dung: Trien khai nhe tren moi truong Python/C++ hoac thiet bi bien co ho tro TFLite.

2. **best_uav_model.onnx (101.45 MB):**
   * Dinh dang: Open Neural Network Exchange (ONNX Opset 17).
   * Dac tinh: Chuan hoa toan bo do thi tinh toan cua CenterNet (Coordinate Attention, P2-FPN, 3 nhanh Head).
   * Muc dich: Dong vai tro cau noi van nang de bien dich sang NVIDIA TensorRT Engine tren Jetson Nano.
   * Lenh bien dich TensorRT Engine FP16 tren Jetson Nano:
     ```bash
     trtexec --onnx=best_uav_model.onnx --saveEngine=best_uav_model_fp16.engine --fp16
     ```

3. **best_uav_model_fp16.keras (102.90 MB):**
   * Dinh dang: Keras 3.x Checkpoint.
   * Dac tinh: Mo hinh Keras ap dung chinh sach mixed precision an toan (Backbone mixed_float16, FPN + Head float32) ngan ngua triet de loi tran so IEEE 754 tren GPU.

## 2. Huong Dan Tai Trong So Mo Hinh Tu Kaggle

Cac tep trong so tren da duoc dong goi va chia se cong khai tai Kaggle Dataset:
* Link Kaggle: https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint

Cach 1: Tai tu dong bang script Python:
```bash
python Model_FP16/download_weights.py
```

Cach 2: Tai bang Kaggle CLI:
```bash
kaggle datasets download -d namnguyen171006/uav-checkpoint -p Model_FP16/ --unzip
```

## 3. Huong Dan Chay Thuc Nghiem va Danh Gia Benchmark

Chay kich ban Python tuong duong so tay Jupyter:
```bash
python Model_FP16/anti_uav_model_fp16.py \
    --voc_root "/duong/dan/den/VOC_AntiUAV_RGB" \
    --val_csv "/duong/dan/den/val_triplets.csv" \
    --checkpoint weights/best_uav_model.keras \
    --batch_size 8 \
    --stride 5 \
    --samples 3000
```
Hoac mo so tay Jupyter `Model_FP16/anti_uav_model_fp16.ipynb` tren moi truong Kaggle/Google Colab/Local GPU.
