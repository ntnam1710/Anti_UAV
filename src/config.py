# -*- coding: utf-8 -*-
"""
Cau hinh cho mo hinh Anti-UAV CenterNet RGB
"""
import os

class Config:
    # INPUT DAU VAO
    INPUT_HEIGHT = 640
    INPUT_WIDTH = 640
    INPUT_CHANNELS = 9
    INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

    ORIG_WIDTH = 1920.0
    ORIG_HEIGHT = 1080.0

    FEATURE_STRIDE = 8
    STRIDE = 8
    GRID_H = INPUT_HEIGHT // FEATURE_STRIDE
    GRID_W = INPUT_WIDTH // FEATURE_STRIDE
    FPN_DIM = 128

    # BIAS TRONG SIGMOID
    # chieu rong drone trong dataset tap trung trong khoang 100-125
    # chieu cao drone trong dataset tap trung khoang 60-75 px
    # w_norm = 125/1920 px va h_norm = 70/1080 ~ 0.065
    # LOGIT BIAS = LN(0.065 / (1-0.065)) ~ -2.66
    SIZE_BIAS_INIT = -2.66

    # XAC SUAT TIEN NGHIEM HEATMAP p0 = 0.01 -> logit bias = ln(0.01 / 0.99) ~ -4.595
    HEATMAP_BIAS_INIT = -4.595

    # LOSS WEIGHTS
    LAMBDA_HEATMAP = 1.0
    LAMBDA_OFFSET = 1.0
    LAMBDA_SIZE = 1.0

    # TRAINING DEFAULTS 
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_LR = 1e-4
    DEFAULT_EPOCHS = 10

    # THAM SO SUY LUAN INFERENCE DEFAULTS
    CONF_THRESH = 0.40
    TOP_K = 100
    NMS_POOL_SIZE = 3

    # THAM SO BOC LOC BAM BAT TRACKER
    TRACKER_ALPHA = 0.6
    TRACKER_MAX_MISSING = 15

    # DEFAULT PATH
    DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "best_uav_model.keras")
    KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/namnguyen171006/anti-uav-rgb"
    KAGGLE_CHECKPOINT_URL = "https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint"
