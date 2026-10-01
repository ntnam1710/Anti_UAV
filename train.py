"""
train model
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import argparse
import pandas as pd 
import tensorflow as tf
from tqdm import tqdm

#import module
from src.config import Config
from src.models import build_detection_model
from src.data import create_temporal_dataset
from src.losses import compute_combined_loss

def setup_hardware():
    #thiet lap cau hinh bo nho va do chinh xac dua tren GPU huan luyen hien tai
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get("device_name", "")
        except Exception:
            gpu_name = gpus[0].name

        print(f"Phat hien GPU: {gpu_name}")
        if "P100" in gpu_name:
            tf.keras.mixed_precision.set_global_policy("float32")
            print("Phat hien Tesla P100: dung mixed_float32 chuan")
        else:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Phat hien GPU khac: dung mixed_float16")
    else:
        print("Khong tim thay GPU, train tren CPU")

def train(voc_root, train_csv, val_csv, checkpoint_dir="weights", epochs=10, batch_size=8, lr=1e-4, resume_weights=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    setup_hardware()

    print("Bat dau huan luyen")
    print(f"So Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Thu muc VOC: {voc_root}")
    print(f"CSV Train: {train_csv}")
    print(f"CSV Val: {val_csv}")
    print(f"Checkpoind: {checkpoint_dir}")
    print("")

    #tien xu ly data
    train_ds, n_train = create_temporal_dataset(
        csv_path=train_csv, voc_root=voc_root,
        batch_size=batch_size, is_training=True
    )
    val_ds, n_val = create_temporal_dataset(
        csv_path=val_csv, voc_root=voc_root,
        batch_size=batch_size, is_training=False
    )

    train_steps = n_train // batch_size
    val_steps = n_val // batch_size

    #build model
    model = build_detection_model(input_shape=Config.INPUT_SHAPE, fpn_dim=Config.FPN_DIM)
    if resume_weights and os.path.exists(resume_weights):
        print(f"Dang nap weight tu: {resume_weights}")
        model.load_weights(resume_weights)
        print("Nap thanh cong")
    else:
        print("Khong tim thay weights")

    #optimizer AdamW
    decay_steps = max(1, epochs * train_steps)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=decay_steps, alpha=0.01 
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    best_val_loss = float("inf")
    history = []

    #train step
    @tf.function
    def train_step(x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            total_loss, hm_loss, sz_loss = compute_combined_loss(y_true, y_pred)
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, hm_loss, sz_loss
    
    #val step
    @tf.function
    def val_step(x, y_true):
        y_pred = model(x, training=False)
        total_loss, hm_loss, sz_loss = compute_combined_loss(y_true, y_pred)
        return total_loss, hm_loss, sz_loss

    #train loop epochs
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"EPOCH {epoch}/{epochs}")

        #train loop
        train_total_loss, train_hm_loss, train_sz_loss = 0.0, 0.0, 0.0
        pbar_train = tqdm(total=train_steps, desc="Train", unit="batch")
        for step, (x_b, y_b) in enumerate(train_ds):
            if step >= train_steps:
                break
            tl, hl, sl = train_step(x_b, y_b)
            train_total_loss += float(tl)
            train_hm_loss += float(hl)
            train_sz_loss += float(sl)
            pbar_train.set_postfix({"loss": f"{tl:.4f}", "hm": f"{hl:.4f}"})
            pbar_train.update(1)
        pbar_train.close()

        train_total_loss /= max(1, train_steps)
        train_hm_loss /= max(1, train_steps)
        train_sz_loss /= max(1, train_steps)

        #val loop
        val_total_loss, val_hm_loss, val_sz_loss = 0.0, 0.0, 0.0
        pbar_val = tqdm(total=val_steps, desc="Val", unit="batch")
        for step, (x_b, y_b) in enumerate(val_ds):
            if step >= val_steps:
                break
            tl, hl, sl = val_step(x_b, y_b)
            val_total_loss += float(tl)
            val_hm_loss += float(hl)
            val_sz_loss += float(sl)
            pbar_val.set_postfix({"loss": f"{tl:.4f}", "hm": f"{hl:.4f}"})
            pbar_val.update(1)
        pbar_val.close()

        val_total_loss /= max(1, val_steps)
        val_hm_loss /= max(1, val_steps)
        val_sz_loss /= max(1, val_steps)

        elapsed_mins = (time.time() - t0) / 60.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Time: {elapsed_mins:.1f} phut:")
        print(f"TRAIN -> Tong Loss: {train_total_loss:.4f} | Heatmap: {train_hm_loss:.4f} | Size/Offset: {train_sz_loss:.4f}")
        print(f"VAL-> Tong Loss: {val_total_loss:.4f} | Heatmap: {val_hm_loss:.4f} | Size/Offset: {val_sz_loss:.4f}")

        #luu checkpoint
        latest_ckpt = os.path.join(checkpoint_dir, "latest_uav_model.keras")
        model.save_weights(latest_ckpt)

        #luu checkpoint tot nhat
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_ckpt = os.path.join(checkpoint_dir, "best_uav_model.keras")
            model.save(best_ckpt)
            print(f"Mo hinh tot nhat (Val Loss: {best_val_loss:.4f})")

        #ghi history train
        history.append({
            "epoch": epoch,
            "train_loss": train_total_loss,
            "train_hm_loss": train_hm_loss,
            "train_sz_loss": train_sz_loss,
            "val_loss": val_total_loss,
            "val_hm_loss": val_hm_loss,
            "val_sz_loss": val_sz_loss,
            "time_mins": elapsed_mins
        })    
        pd.DataFrame(history).to_csv(
            os.path.join(checkpoint_dir, "training_history.csv"), index = False
        )

    print("\n Train Model Complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen Anti_UAV CenterNet RGB")
    parser.add_argument("--voc_root", "--vocroot", dest="voc_root", type=str, required=True, help="Thu muc goc dataset VOC")
    parser.add_argument("--train_csv", type=str, required=True, help="Tep CSV danh sach Train")
    parser.add_argument("--val_csv", type=str, required=True, help="Tep CSV danh sach Val")
    parser.add_argument("--checkpoint_dir", type=str, default="weights", help="Thu muc luu tru checkpoint")
    parser.add_argument("--epochs", type=int, default=Config.DEFAULT_EPOCHS, help="So epochs train")
    parser.add_argument("--batch_size", type=int, default=Config.DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.DEFAULT_LR, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Duong dan file weight de tiep tuc train")

    args = parser.parse_args()
    
    train(
        voc_root=args.voc_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_weights=args.resume
    )