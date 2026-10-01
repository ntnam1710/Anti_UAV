# -*- coding: utf-8 -*-
"""
Kịch bản hỗ trợ tự động tải trọng số tiền huấn luyện best_uav_model.keras từ Kaggle.
"""
import os
import sys
import subprocess

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.path.join(WEIGHTS_DIR, "best_uav_model.keras")

def download():
    """
    Tải trọng số mô hình từ Kaggle Dataset bằng Kaggle CLI nếu tệp chưa tồn tại.
    """
    if os.path.exists(TARGET_PATH) and os.path.getsize(TARGET_PATH) > 50 * 1024 * 1024:
        file_size_mb = os.path.getsize(TARGET_PATH) / (1024 * 1024)
        print(f"[XAC NHAN] Trong so tien huan luyen da ton tai tai: {TARGET_PATH} ({file_size_mb:.2f} MB)")
        return

    print("[INFO] Dang tai trong so tu Kaggle Dataset: namnguyen171006/uav-checkpoint...")
    try:
        cmd = f"kaggle datasets download -d namnguyen171006/uav-checkpoint -p \"{WEIGHTS_DIR}\" --unzip"
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(TARGET_PATH):
            file_size_mb = os.path.getsize(TARGET_PATH) / (1024 * 1024)
            print(f"[HOAN TAT] Da tai trong so thanh cong ve: {TARGET_PATH} ({file_size_mb:.2f} MB)")
        else:
            print("[CANH BAO] Qua trinh tai hoan tat nhung khong tim thay best_uav_model.keras trong thu muc dich.")
    except Exception as e:
        print(f"[LOI] Khong the tai tu dong qua Kaggle CLI: {e}")
        print("Vui long tai thu cong tu dia chi: https://www.kaggle.com/datasets/namnguyen171006/uav-checkpoint")

if __name__ == "__main__":
    download()
