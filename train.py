# -*- coding: utf-8 -*-
"""
Training Pipeline for Anti-UAV CenterNet RGB
Supports:
- 9-channel Visible RGB temporal triplets (t-1, t, t+1)
- Mixed precision training / float32 on Tesla P100
- Multi-task Anchor-Free loss (Gaussian Focal Heatmap + Offset L1 + Size L1/NWD)
- Automatic checkpointing and training history logging
"""
import os
import sys
import time
import argparse
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.config import Config
from src.models import build_detection_model
from src.data import create_temporal_dataset
from src.losses import compute_combined_loss

def setup_hardware():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get("device_name", "")
        except Exception:
            gpu_name = gpus[0].name

        print(f">> Detected GPU: {gpu_name}")
        if "P100" in gpu_name:
            tf.keras.mixed_precision.set_global_policy("float32")
            print(">> Tesla P100 detected: Using float32 policy.")
        else:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print(">> Tensor Cores detected: Using mixed_float16 policy.")
    else:
        print("[!] No GPU detected, running on CPU.")

def train(voc_root, train_csv, val_csv, checkpoint_dir="weights",
          epochs=10, batch_size=8, lr=1e-4, resume_weights=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    setup_hardware()

    print("=" * 70)
    print(" STARTING TRAINING: ANTI-UAV ANCHOR-FREE CENTERNET RGB")
    print(f" Epochs          : {epochs}")
    print(f" Batch Size      : {batch_size}")
    print(f" Learning Rate   : {lr}")
    print(f" VOC Root        : {voc_root}")
    print(f" Train CSV       : {train_csv}")
    print(f" Val CSV         : {val_csv}")
    print(f" Checkpoint Dir  : {checkpoint_dir}")
    print("=" * 70)

    # 1. DataLoader
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

    # 2. Build Model
    model = build_detection_model(input_shape=Config.INPUT_SHAPE, fpn_dim=Config.FPN_DIM)
    if resume_weights and os.path.exists(resume_weights):
        print(f">> Loading pretrained weights from: {resume_weights}")
        model.load_weights(resume_weights)

    # 3. Optimizer & LR Scheduler
    decay_steps = max(1, epochs * train_steps)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=decay_steps, alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    best_val_loss = float("inf")
    history_records = []

    # Training Step
    @tf.function
    def train_step(x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            total_loss, hm_loss, sz_loss = compute_combined_loss(y_true, y_pred)
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, hm_loss, sz_loss

    # Validation Step
    @tf.function
    def val_step(x, y_true):
        y_pred = model(x, training=False)
        total_loss, hm_loss, sz_loss = compute_combined_loss(y_true, y_pred)
        return total_loss, hm_loss, sz_loss

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n>> Epoch [{epoch:02d}/{epochs:02d}]")

        # Train loop
        train_tot_loss, train_hm_loss, train_sz_loss = 0.0, 0.0, 0.0
        pbar_train = tqdm(total=train_steps, desc="Train", unit="batch")
        for step, (x_b, y_b) in enumerate(train_ds):
            if step >= train_steps:
                break
            tl, hl, sl = train_step(x_b, y_b)
            train_tot_loss += float(tl)
            train_hm_loss += float(hl)
            train_sz_loss += float(sl)
            pbar_train.set_postfix({"loss": f"{tl:.4f}", "hm": f"{hl:.4f}"})
            pbar_train.update(1)
        pbar_train.close()

        train_tot_loss /= max(1, train_steps)
        train_hm_loss /= max(1, train_steps)
        train_sz_loss /= max(1, train_steps)

        # Validation loop
        val_tot_loss, val_hm_loss, val_sz_loss = 0.0, 0.0, 0.0
        pbar_val = tqdm(total=val_steps, desc="Val", unit="batch")
        for step, (x_b, y_b) in enumerate(val_ds):
            if step >= val_steps:
                break
            tl, hl, sl = val_step(x_b, y_b)
            val_tot_loss += float(tl)
            val_hm_loss += float(hl)
            val_sz_loss += float(sl)
            pbar_val.set_postfix({"val_loss": f"{tl:.4f}"})
            pbar_val.update(1)
        pbar_val.close()

        val_tot_loss /= max(1, val_steps)
        val_hm_loss /= max(1, val_steps)
        val_sz_loss /= max(1, val_steps)

        elapsed_mins = (time.time() - t0) / 60.0

        print(f">> Epoch [{epoch:02d}/{epochs:02d}] Finished in {elapsed_mins:.1f} mins:")
        print(f"   TRAIN -> Total: {train_tot_loss:.4f} | Heatmap: {train_hm_loss:.4f} | Size: {train_sz_loss:.4f}")
        print(f"   VAL   -> Total: {val_tot_loss:.4f} | Heatmap: {val_hm_loss:.4f} | Size: {val_sz_loss:.4f}")

        # Save checkpoint
        latest_ckpt = os.path.join(checkpoint_dir, "latest_uav_model.keras")
        model.save(latest_ckpt)

        if val_tot_loss < best_val_loss:
            best_val_loss = val_tot_loss
            best_ckpt = os.path.join(checkpoint_dir, "best_uav_model.keras")
            model.save(best_ckpt)
            print(f"   ⭐ NEW BEST MODEL SAVED! (Val Loss: {best_val_loss:.4f})")

        # Log history
        history_records.append({
            "epoch": epoch,
            "train_loss": train_tot_loss,
            "train_hm_loss": train_hm_loss,
            "train_size_loss": train_sz_loss,
            "val_loss": val_tot_loss,
            "val_hm_loss": val_hm_loss,
            "val_size_loss": val_sz_loss,
            "time_minutes": round(elapsed_mins, 2)
        })
        pd.DataFrame(history_records).to_csv(
            os.path.join(checkpoint_dir, "training_history.csv"), index=False
        )

    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_root", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
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
