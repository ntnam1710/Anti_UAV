# -*- coding: utf-8 -*-
"""
Central Configuration for Anti-UAV RGB CenterNet
"""
import os

class Config:
    # Model architecture
    INPUT_HEIGHT = 640
    INPUT_WIDTH = 640
    INPUT_CHANNELS = 9  # 3-frame temporal triplet (t-1, t, t+1) x 3 RGB channels
    INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
    
    FEATURE_STRIDE = 8  # P2 resolution: 640 / 8 = 80x80
    GRID_H = INPUT_HEIGHT // FEATURE_STRIDE
    GRID_W = INPUT_WIDTH // FEATURE_STRIDE
    FPN_DIM = 128
    
    # Prior initialization
    # Drone average size ~ 0.065 of image -> logit is ln(0.065 / (1 - 0.065)) ~ -2.66
    SIZE_BIAS_INIT = -2.66
    # Heatmap background prior p0 = 0.1 -> logit is ln(0.1 / 0.9) ~ -2.19
    HEATMAP_BIAS_INIT = -2.19
    
    # Loss weights
    LAMBDA_HEATMAP = 1.0
    LAMBDA_SIZE = 0.1
    LAMBDA_OFFSET = 1.0
    
    # Training defaults
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_LR = 1e-4
    DEFAULT_EPOCHS = 10
    
    # Inference defaults
    CONF_THRESH = 0.20
    TOP_K = 100
    NMS_POOL_SIZE = 3
    
    # Tracker defaults
    TRACKER_ALPHA = 0.6
    TRACKER_MAX_MISSING = 15
    
    # Default Paths
    DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "best_uav_model.keras")
    KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/namnguyen171006/anti-uav-rgb"
    KAGGLE_CHECKPOINT_URL = "https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint"
