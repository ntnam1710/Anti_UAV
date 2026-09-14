# 🎯 Pretrained Weights: best_uav_model.keras

This folder contains or tracks the optimal trained model weights for the **Anti-UAV Anchor-Free CenterNet (RGB-Only)** model.

## Model Specifications
* **Architecture**: ResNet-50v2 Backbone + Coordinate Attention + P2-FPN + Decoupled CenterNet Heads
* **Input Shape**: `(640, 640, 9)` (3-frame Visible RGB temporal triplet: $t-1, t, t+1$)
* **Size Head Activation**: Sigmoid with Logit Prior Initialization `Constant(-2.66)`
* **File Size**: ~102.8 MB
* **Format**: Keras SavedModel 3.x (`.keras`)

## How to Download Weights
If `best_uav_model.keras` is not present locally (due to GitHub's 100MB file limit), you can download it using any of the following methods:

### Method 1: Download from Kaggle Dataset
The weights are hosted on Kaggle:
👉 **[Kaggle Dataset: uav-checkpoint](https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint)**

Place the downloaded `best_uav_model.keras` directly into this `weights/` folder.

### Method 2: Python Script (Automatic)
Run the automated downloader:
```bash
python weights/download_weights.py
```

### Method 3: Using Kaggle CLI
```bash
kaggle datasets download -d namnguyen171006/uav-checkpoint -p weights/ --unzip
```
