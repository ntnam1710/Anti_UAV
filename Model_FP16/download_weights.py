# -*- coding: utf-8 -*-
"""
Kich ban ho tro tu dong tai cac tep trong so moi (TFLite FP16, ONNX, Keras FP16) tu Kaggle Dataset.
Dataset: namnguyen171006/uav-checkpoint
"""

import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_TO_CHECK = [
    "best_uav_model_fp16.tflite",
    "best_uav_model.onnx",
    "best_uav_model_fp16.keras"
]

def download_models():
    all_exist = all(os.path.exists(os.path.join(CURRENT_DIR, f)) for f in FILES_TO_CHECK)
    if all_exist:
        print("[XAC NHAN] Toan bo cac file trong so Model_FP16 da ton tai san tren may.")
        return

    print("[INFO] Dang tai trong so moi tu Kaggle Dataset: namnguyen171006/uav-checkpoint...")
    try:
        cmd = f'kaggle datasets download -d namnguyen171006/uav-checkpoint -p "{CURRENT_DIR}" --unzip'
        subprocess.run(cmd, shell=True, check=True)
        print("[HOAN TAT] Da tai va giai nen thanh cong cac trong so moi vao Model_FP16.")
    except Exception as e:
        print(f"[LOI] Khong the tai tu dong qua Kaggle CLI: {e}")
        print("Vui long tai thu cong tu dia chi: https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint")

if __name__ == "__main__":
    download_models()
