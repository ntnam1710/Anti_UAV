# -*- coding: utf-8 -*-
"""
Helper script to download pretrained weights best_uav_model.keras from Kaggle.
"""
import os
import sys
import subprocess

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.path.join(WEIGHTS_DIR, "best_uav_model.keras")

def download():
    if os.path.exists(TARGET_PATH) and os.path.getsize(TARGET_PATH) > 50 * 1024 * 1024:
        print(f"[OK] Pretrained weights already exist at: {TARGET_PATH} ({os.path.getsize(TARGET_PATH)/(1024*1024):.2f} MB)")
        return

    print(">> Downloading weights from Kaggle dataset: namnguyen171006/uav-checkpoint...")
    try:
        cmd = f"kaggle datasets download -d namnguyen171006/uav-checkpoint -p \"{WEIGHTS_DIR}\" --unzip"
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(TARGET_PATH):
            print(f"✅ Successfully downloaded weights to: {TARGET_PATH}")
        else:
            print("[!] Download finished but best_uav_model.keras was not found in destination.")
    except Exception as e:
        print(f"[ERROR] Could not download automatically via Kaggle CLI: {e}")
        print("Please manually download from: https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint")

if __name__ == "__main__":
    download()
