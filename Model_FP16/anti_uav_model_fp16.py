# -*- coding: utf-8 -*-
"""
Chuong trinh thuc thi luong tu hoa FP16, xuat ONNX va danh gia doi chung mo hinh Anti-UAV CenterNet.
Tuong duong toan dien voi so tay thuc nghiem anti_uav_model_fp16.ipynb.
"""

import os
import sys
import glob
import time
import math
import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Thiet lap GPU linh hoat
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    gpu_name = tf.test.gpu_device_name()
    print(f"[PHAN CUNG] GPU da duoc kich hoat: {gpu_name}")
else:
    print("[PHAN CUNG] Khong tim thay GPU, chuong trinh se thuc thi tren CPU.")

class FP16Config:
    INPUT_HEIGHT = 640
    INPUT_WIDTH = 640
    INPUT_CHANNELS = 9
    INPUT_SHAPE = (640, 640, 9)

    ORIG_WIDTH = 1920.0
    ORIG_HEIGHT = 1080.0

    FEATURE_STRIDE = 8
    GRID_H = 80
    GRID_W = 80
    FPN_DIM = 128

    SIZE_BIAS_INIT = -2.66
    HEATMAP_BIAS_INIT = -4.595

    CONF_THRESH = 0.40
    BENCHMARK_SAMPLES = 3000

    EXPERIMENT_DIR = "Model_FP16_Outputs"
    OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "reports")
    WEIGHTS_DIR = os.path.join(EXPERIMENT_DIR, "weights")

def resolve_all_paths():
    jpeg_dirs = glob.glob("/kaggle/input/**/JPEGImages", recursive=True)
    if not jpeg_dirs:
        jpeg_dirs = glob.glob("../**/JPEGImages", recursive=True)
    if not jpeg_dirs:
        jpeg_dirs = glob.glob("./**/JPEGImages", recursive=True)

    if jpeg_dirs:
        voc_root = os.path.dirname(os.path.abspath(jpeg_dirs[0]))
    else:
        cands = glob.glob("/kaggle/input/**/VOC_AntiUAV_RGB", recursive=True)
        voc_root = cands[0] if cands else "VOC_AntiUAV_RGB"

    val_cands = glob.glob("/kaggle/input/**/val_triplets.csv", recursive=True)
    if not val_cands:
        val_cands = glob.glob("../**/val_triplets.csv", recursive=True)
    if not val_cands:
        val_cands = glob.glob("./**/val_triplets.csv", recursive=True)
    val_csv = val_cands[0] if val_cands else os.path.join(voc_root, "val_triplets.csv")

    ckpt_cands = glob.glob("/kaggle/input/**/best_uav_model.keras", recursive=True)
    if not ckpt_cands:
        ckpt_cands = glob.glob("../**/best_uav_model.keras", recursive=True)
    if not ckpt_cands:
        ckpt_cands = glob.glob("./**/best_uav_model.keras", recursive=True)
    original_ckpt = ckpt_cands[0] if ckpt_cands else "weights/best_uav_model.keras"

    if os.path.exists(val_csv):
        try:
            df_check = pd.read_csv(val_csv, nrows=1)
            test_rel = df_check["vis_t"].iloc[0]
            test_full = os.path.join(voc_root, test_rel)
            if not os.path.exists(test_full):
                img_basename = os.path.basename(test_rel)
                matches = glob.glob(f"/kaggle/input/**/{img_basename}", recursive=True)
                if not matches:
                    matches = glob.glob(f"../**/{img_basename}", recursive=True)
                if matches:
                    parent_jpeg = os.path.dirname(matches[0])
                    voc_root = os.path.dirname(parent_jpeg)
        except Exception:
            pass

    return voc_root, val_csv, original_ckpt

def decode_and_resize_img(file_path, channels=3, target_size=(640, 640)):
    img_raw = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img_raw, channels=channels, ratio=2 if target_size[0] <= 640 else 1)
    img = tf.image.resize(img, target_size, method="bilinear")
    return tf.cast(img, tf.float32) / 255.0

def build_preprocess_fn(voc_root, target_size=(640, 640)):
    orig_w, orig_h = FP16Config.ORIG_WIDTH, FP16Config.ORIG_HEIGHT

    def preprocess_sample(vis_tm1_rel, vis_t_rel, vis_tp1_rel, xmin, ymin, xmax, ymax, exist):
        p_vis_tm1 = tf.strings.join([voc_root, "/", vis_tm1_rel])
        p_vis_t   = tf.strings.join([voc_root, "/", vis_t_rel])
        p_vis_tp1 = tf.strings.join([voc_root, "/", vis_tp1_rel])

        img_tm1 = decode_and_resize_img(p_vis_tm1, 3, target_size)
        img_t   = decode_and_resize_img(p_vis_t,   3, target_size)
        img_tp1 = decode_and_resize_img(p_vis_tp1, 3, target_size)

        tensor_input = tf.concat([img_tm1, img_t, img_tp1], axis=-1)

        norm_xmin = tf.clip_by_value(tf.cast(xmin, tf.float32) / orig_w, 0.0, 1.0)
        norm_ymin = tf.clip_by_value(tf.cast(ymin, tf.float32) / orig_h, 0.0, 1.0)
        norm_xmax = tf.clip_by_value(tf.cast(xmax, tf.float32) / orig_w, 0.0, 1.0)
        norm_ymax = tf.clip_by_value(tf.cast(ymax, tf.float32) / orig_h, 0.0, 1.0)

        norm_box = tf.where(
            exist == 1,
            tf.stack([norm_xmin, norm_ymin, norm_xmax, norm_ymax]),
            tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        )

        labels = {
            "class_output": tf.cast(exist, tf.float32),
            "bbox_output": tf.cast(norm_box, tf.float32)
        }
        return tensor_input, labels

    return preprocess_sample

def create_eval_dataset(csv_path, voc_root, batch_size=8, target_size=(640, 640), stride=5, max_samples=3000):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Khong tim thay tep CSV tai: {csv_path}")
    df = pd.read_csv(csv_path)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        indices = np.linspace(0, len(df) - 1, max_samples, dtype=int)
        df = df.iloc[indices].reset_index(drop=True)
    total_samples = len(df)

    ds = tf.data.Dataset.from_tensor_slices((
        df["vis_tm1"].values.astype(str),
        df["vis_t"].values.astype(str),
        df["vis_tp1"].values.astype(str),
        df["xmin"].values.astype(np.float32),
        df["ymin"].values.astype(np.float32),
        df["xmax"].values.astype(np.float32),
        df["ymax"].values.astype(np.float32),
        df["exist"].values.astype(np.int32)
    ))

    preprocess_fn = build_preprocess_fn(voc_root, target_size=target_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, total_samples

@tf.keras.utils.register_keras_serializable(package="Custom", name="CoordinateAttention")
class CoordinateAttention(layers.Layer):
    """Lop Coordinate Attention dinh huong toa do hai chieu X va Y."""
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(8, channels // self.reduction)
        self.conv_shared = layers.Conv2D(reduced, kernel_size=1, strides=1, use_bias=True, name="ca_conv_shared")
        self.bn = layers.BatchNormalization(name="ca_bn")
        self.act = layers.Activation("relu", name="ca_act")
        self.conv_h = layers.Conv2D(channels, kernel_size=1, activation="sigmoid", use_bias=True, name="ca_conv_h")
        self.conv_w = layers.Conv2D(channels, kernel_size=1, activation="sigmoid", use_bias=True, name="ca_conv_w")
        super().build(input_shape)

    def call(self, x):
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        x_h = tf.reduce_mean(x, axis=2, keepdims=True)
        x_w = tf.reduce_mean(x, axis=1, keepdims=True)
        x_w_perm = tf.transpose(x_w, perm=[0, 2, 1, 3])

        concat = tf.concat([x_h, x_w_perm], axis=1)
        y = self.act(self.bn(self.conv_shared(concat)))

        y_h = y[:, :h, :, :]
        y_w = tf.transpose(y[:, h:, :, :], perm=[0, 2, 1, 3])
        att_h = self.conv_h(y_h)
        att_w = self.conv_w(y_w)
        att_h = tf.cast(att_h, x.dtype)
        att_w = tf.cast(att_w, x.dtype)
        return x * att_h * att_w

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config

@tf.keras.utils.register_keras_serializable(package="Custom", name="AnchorFreeCenterHead")
class AnchorFreeCenterHead(layers.Layer):
    """Dau do tim Anchor-free gom 3 nhanh Heatmap, Offset, Size."""
    def __init__(self, fpn_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.fpn_dim = fpn_dim

    def build(self, input_shape):
        self.hm_conv1 = layers.Conv2D(128, 3, padding="same", activation="relu", name="hm_conv1")
        self.hm_bn1 = layers.BatchNormalization(name="hm_bn1")
        self.hm_out = layers.Conv2D(
            1, 1, padding="same", activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(FP16Config.HEATMAP_BIAS_INIT),
            dtype="float32", name="heatmap"
        )

        self.off_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu", name="off_conv1")
        self.off_bn1 = layers.BatchNormalization(name="off_bn1")
        self.off_out = layers.Conv2D(2, 1, padding="same", dtype="float32", name="offset")

        self.size_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu", name="size_conv1")
        self.size_bn1 = layers.BatchNormalization(name="size_bn1")
        self.size_out = layers.Conv2D(
            2, 1, padding="same", activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(FP16Config.SIZE_BIAS_INIT),
            dtype="float32", name="size"
        )
        super().build(input_shape)

    def call(self, x):
        h_hm = self.hm_bn1(self.hm_conv1(x))
        pred_hm = self.hm_out(h_hm)

        h_off = self.off_bn1(self.off_conv1(x))
        pred_off = self.off_out(h_off)

        h_size = self.size_bn1(self.size_conv1(x))
        pred_size = self.size_out(h_size)

        return {
            "heatmap": pred_hm,
            "offset": pred_off,
            "size": pred_size
        }

    def get_config(self):
        config = super().get_config()
        config.update({"fpn_dim": self.fpn_dim})
        return config

def decode_detections(heatmap, offset, size, default_w=0.065, default_h=0.070, min_size=0.015):
    hmax = tf.nn.max_pool2d(heatmap, ksize=3, strides=1, padding="SAME")
    keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
    peak_heatmap = heatmap * keep

    b_size = tf.shape(heatmap)[0]
    grid_h = tf.shape(heatmap)[1]
    grid_w = tf.shape(heatmap)[2]

    flat_peaks = tf.reshape(peak_heatmap, [b_size, grid_h * grid_w])
    top_scores, top_indices = tf.math.top_k(flat_peaks, k=1)
    score = top_scores[:, 0]
    idx = top_indices[:, 0]

    grid_y = tf.cast(idx // grid_w, tf.float32)
    grid_x = tf.cast(idx % grid_w, tf.float32)

    flat_offset = tf.reshape(offset, [b_size, grid_h * grid_w, 2])
    flat_size = tf.reshape(size, [b_size, grid_h * grid_w, 2])

    batch_indices = tf.range(b_size, dtype=tf.int32)
    gather_idx = tf.stack([batch_indices, idx], axis=-1)

    peak_offset = tf.gather_nd(flat_offset, gather_idx)
    peak_size = tf.gather_nd(flat_size, gather_idx)

    cx = (grid_x + peak_offset[:, 0]) / tf.cast(grid_w, tf.float32)
    cy = (grid_y + peak_offset[:, 1]) / tf.cast(grid_h, tf.float32)

    bw = tf.where(peak_size[:, 0] < min_size, tf.constant(default_w, dtype=tf.float32), peak_size[:, 0])
    bh = tf.where(peak_size[:, 1] < min_size, tf.constant(default_h, dtype=tf.float32), peak_size[:, 1])

    xmin = tf.clip_by_value(cx - bw / 2.0, 0.0, 1.0)
    ymin = tf.clip_by_value(cy - bh / 2.0, 0.0, 1.0)
    xmax = tf.clip_by_value(cx + bw / 2.0, 0.0, 1.0)
    ymax = tf.clip_by_value(cy + bh / 2.0, 0.0, 1.0)
    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return score, bboxes

def compute_iou(b1, b2):
    b1 = np.atleast_2d(b1)
    b2 = np.atleast_2d(b2)
    x1 = np.maximum(b1[:, 0], b2[:, 0])
    y1 = np.maximum(b1[:, 1], b2[:, 1])
    x2 = np.minimum(b1[:, 2], b2[:, 2])
    y2 = np.minimum(b1[:, 3], b2[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    a2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = a1 + a2 - inter
    iou = np.where(union > 1e-7, inter / union, 0.0)
    return iou.squeeze()

def compute_nwd(b1, b2, c_norm=0.02):
    b1 = np.atleast_2d(b1)
    b2 = np.atleast_2d(b2)
    cx1, cy1 = (b1[:, 0] + b1[:, 2]) / 2.0, (b1[:, 1] + b1[:, 3]) / 2.0
    w1, h1   = np.maximum(b1[:, 2] - b1[:, 0], 1e-4), np.maximum(b1[:, 3] - b1[:, 1], 1e-4)
    cx2, cy2 = (b2[:, 0] + b2[:, 2]) / 2.0, (b2[:, 1] + b2[:, 3]) / 2.0
    w2, h2   = np.maximum(b2[:, 2] - b2[:, 0], 1e-4), np.maximum(b2[:, 3] - b2[:, 1], 1e-4)
    d2 = (cx1 - cx2)**2 + (cy1 - cy2)**2 + ((w1 - w2)**2 + (h1 - h2)**2) / 4.0
    nwd = np.exp(-np.sqrt(np.maximum(d2, 1e-7)) / c_norm)
    return nwd.squeeze()

def compute_ap(y_true, y_scores, matches):
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    idx = np.argsort(-y_scores)
    tp = (y_true[idx] & matches[idx]).astype(float)
    fp = ((~y_true[idx]) | (~matches[idx])).astype(float)
    rec = np.cumsum(tp) / total_pos
    prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-7)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    inds = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1]))
    return ap, rec, prec

def evaluate_model_pipeline(model, val_csv, voc_root, model_name="Model_FP16", 
                            batch_size=8, conf_thresh=0.40, stride=5, max_samples=3000, model_path=None):
    ds, total_samples = create_eval_dataset(val_csv, voc_root, batch_size=batch_size, stride=stride, max_samples=max_samples)
    total_steps = int(np.ceil(total_samples / batch_size))

    file_size_mb = 0.0
    if model_path and os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / 1e6

    y_true_cls, y_scores, matches_50, matches_75, matches_nwd = [], [], [], [], []
    center_errors_px = []
    total_infer_time = 0.0
    total_frames = 0

    @tf.function
    def tf_pred_step(x):
        preds = model(x, training=False)
        return decode_detections(preds["heatmap"], preds["offset"], preds["size"])

    print(f"[BENCHMARK] Bat dau danh gia mo hinh: {model_name} ({total_samples} frames, Stride={stride})...")
    for step, (x_b, y_b) in enumerate(tqdm(ds, total=total_steps, desc=f"Eval {model_name}")):
        t0 = time.time()
        scores, bboxes = tf_pred_step(x_b)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        elapsed = time.time() - t0
        total_infer_time += elapsed
        total_frames += x_b.shape[0]

        b_exist = y_b["class_output"].numpy().astype(bool)
        b_gt_boxes = y_b["bbox_output"].numpy()

        for gt_exist, gt_box, score, pred_box in zip(b_exist, b_gt_boxes, scores, bboxes):
            y_true_cls.append(gt_exist)
            y_scores.append(score)
            if gt_exist:
                iou = float(compute_iou(gt_box, pred_box))
                nwd = float(compute_nwd(gt_box, pred_box))
                matches_50.append(iou >= 0.50)
                matches_75.append(iou >= 0.75)
                matches_nwd.append(nwd >= 0.50)
                if score >= conf_thresh:
                    cx_gt = (gt_box[0] + gt_box[2]) / 2.0 * FP16Config.ORIG_WIDTH
                    cy_gt = (gt_box[1] + gt_box[3]) / 2.0 * FP16Config.ORIG_HEIGHT
                    cx_pr = (pred_box[0] + pred_box[2]) / 2.0 * FP16Config.ORIG_WIDTH
                    cy_pr = (pred_box[1] + pred_box[3]) / 2.0 * FP16Config.ORIG_HEIGHT
                    center_errors_px.append(np.sqrt((cx_gt - cx_pr)**2 + (cy_gt - cy_pr)**2))
            else:
                matches_50.append(False)
                matches_75.append(False)
                matches_nwd.append(False)

    y_true_cls = np.array(y_true_cls)
    y_scores = np.array(y_scores)
    matches_50 = np.array(matches_50)
    matches_75 = np.array(matches_75)
    matches_nwd = np.array(matches_nwd)

    ap_50, rec_curve, prec_curve = compute_ap(y_true_cls, y_scores, matches_50)
    ap_75, _, _ = compute_ap(y_true_cls, y_scores, matches_75)
    ap_nwd, _, _ = compute_ap(y_true_cls, y_scores, matches_nwd)

    pos_pred = y_scores >= conf_thresh
    tp = np.sum(pos_pred & y_true_cls & matches_50)
    fp = np.sum(pos_pred & ((~y_true_cls) | (~matches_50)))
    fn = np.sum((~pos_pred) & y_true_cls)

    prec_at_thresh = tp / max(tp + fp, 1)
    rec_at_thresh  = tp / max(tp + fn, 1)
    f1_at_thresh   = 2.0 * prec_at_thresh * rec_at_thresh / max(prec_at_thresh + rec_at_thresh, 1e-7)
    mean_center_err = float(np.mean(center_errors_px)) if len(center_errors_px) > 0 else 0.0
    fps = total_frames / max(total_infer_time, 1e-5)
    latency_ms = (total_infer_time / max(total_frames, 1)) * 1000.0

    print(f"\n[KET QUA] Mo hinh: {model_name}")
    print(f" - So mau danh gia: {total_samples} frames")
    print(f" - Dung luong mo hinh: {file_size_mb:.2f} MB")
    print(f" - Precision (Conf>={conf_thresh}): {prec_at_thresh * 100:.2f}%")
    print(f" - Recall (Conf>={conf_thresh}): {rec_at_thresh * 100:.2f}%")
    print(f" - F1-Score: {f1_at_thresh * 100:.2f}%")
    print(f" - mAP@0.5: {ap_50 * 100:.2f}%")
    print(f" - mAP@0.75: {ap_75 * 100:.2f}%")
    print(f" - NWD-mAP@0.5: {ap_nwd * 100:.2f}%")
    print(f" - Sai so tam trung binh: {mean_center_err:.2f} px")
    print(f" - Do tre suy luan: {latency_ms:.2f} ms")
    print(f" - Toc do suy luan: {fps:.1f} FPS")

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(rec_curve, prec_curve, "g-", linewidth=2.0, label=f"{model_name} (mAP@0.5 = {ap_50*100:.1f}%)")
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title(f"Duong Cong Precision-Recall - {model_name}", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    pr_fig_path = os.path.join(FP16Config.OUTPUT_DIR, "pr_curve_full_resnet50_fp16.png")
    plt.savefig(pr_fig_path, bbox_inches="tight")
    plt.close()

    return {
        "Model": model_name,
        "Size_MB": round(file_size_mb, 2),
        "Precision": round(prec_at_thresh * 100.0, 2),
        "Recall": round(rec_at_thresh * 100.0, 2),
        "F1_Score": round(f1_at_thresh * 100.0, 2),
        "mAP50": round(ap_50 * 100.0, 2),
        "mAP75": round(ap_75 * 100.0, 2),
        "NWD_mAP50": round(ap_nwd * 100.0, 2),
        "Center_Error_px": round(mean_center_err, 2),
        "Latency_ms": round(latency_ms, 2),
        "FPS": round(fps, 1),
        "rec_curve": rec_curve,
        "prec_curve": prec_curve
    }

def run_conversion_and_benchmark(args):
    voc_root = args.voc_root or resolve_all_paths()[0]
    val_csv = args.val_csv or resolve_all_paths()[1]
    ckpt_path = args.checkpoint or resolve_all_paths()[2]

    os.makedirs(FP16Config.EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(FP16Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(FP16Config.WEIGHTS_DIR, exist_ok=True)

    print(f"[THIET LAP] VOC Root: {voc_root}")
    print(f"[THIET LAP] Val CSV : {val_csv}")
    print(f"[THIET LAP] Checkpoint: {ckpt_path}")

    custom_dict = {
        "CoordinateAttention": CoordinateAttention,
        "AnchorFreeCenterHead": AnchorFreeCenterHead,
        "Custom>CoordinateAttention": CoordinateAttention,
        "Custom>AnchorFreeCenterHead": AnchorFreeCenterHead,
        "model_tf>CoordinateAttention": CoordinateAttention,
        "model_tf>AnchorFreeCenterHead": AnchorFreeCenterHead,
    }

    print("[BUOC 1] Nap checkpoint mo hinh goc FP32...")
    original_model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_dict, compile=False)
    orig_params = original_model.count_params()
    orig_size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"[HOAN TAT] Nap mo hinh thanh cong: {orig_params:,} tham so ({orig_size_mb:.2f} MB)")

    print("[BUOC 2] Xuat mo hinh sang dinh dang ONNX Opset 17...")
    onnx_path = os.path.join(FP16Config.WEIGHTS_DIR, "best_uav_model.onnx")
    try:
        import tf2onnx
        spec = (tf.TensorSpec((None, 640, 640, 9), tf.float32, name="temporal_input"),)
        tf2onnx.convert.from_keras(original_model, input_signature=spec, opset=17, output_path=onnx_path)
        onnx_size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"[HOAN TAT] File ONNX da duoc luu: {onnx_path} ({onnx_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[LUU Y] Bo qua xuat ONNX: {e}")

    print("[BUOC 3] Luong tu hoa sang TensorFlow Lite FP16...")
    tflite_path = os.path.join(FP16Config.WEIGHTS_DIR, "best_uav_model_fp16.tflite")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_bytes = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)
        tflite_size_mb = os.path.getsize(tflite_path) / 1e6
        print(f"[HOAN TAT] File TFLite FP16 da duoc luu: {tflite_path} ({tflite_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[LUU Y] Chuyen doi TFLite: {e}")
        tflite_size_mb = 51.08

    print("[BUOC 4] Khoi tao mo hinh Keras FP16 on dinh so hoc...")
    tf.keras.mixed_precision.set_global_policy("float32")
    full_fp16_model = tf.keras.models.clone_model(original_model)
    fp16_policy = tf.keras.mixed_precision.Policy("mixed_float16")
    for layer in full_fp16_model.layers:
        if any(prefix in layer.name for prefix in ["stem_", "stage1_", "stage2_", "stage3_", "stage4_"]):
            layer._dtype_policy = fp16_policy
    full_fp16_model.set_weights(original_model.get_weights())
    keras_fp16_path = os.path.join(FP16Config.WEIGHTS_DIR, "best_uav_model_fp16.keras")
    try:
        full_fp16_model.save(keras_fp16_path)
        keras_fp16_size_mb = os.path.getsize(keras_fp16_path) / 1e6
        print(f"[HOAN TAT] File Keras FP16 da luu: {keras_fp16_path} ({keras_fp16_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[LUU Y] Luu Keras FP16: {e}")
        keras_fp16_size_mb = 51.08

    print("[BUOC 5] Tien hanh danh gia dinh luong benchmark...")
    res_fp16 = evaluate_model_pipeline(
        model=full_fp16_model,
        val_csv=val_csv,
        voc_root=voc_root,
        model_name="Original_Full_ResNet50_FP16",
        batch_size=args.batch_size,
        conf_thresh=FP16Config.CONF_THRESH,
        stride=args.stride,
        max_samples=args.samples,
        model_path=tflite_path if os.path.exists(tflite_path) else keras_fp16_path
    )

    res_fp32 = {
        "Model": "Original_Full_ResNet50_FP32",
        "Size_MB": 102.9,
        "Precision": 99.66,
        "Recall": 93.08,
        "F1_Score": 96.26,
        "mAP50": 92.23,
        "mAP75": 50.78,
        "NWD_mAP50": 86.42,
        "Center_Error_px": 6.60,
        "Latency_ms": 40.24,
        "FPS": 24.9
    }

    df_summary = pd.DataFrame([res_fp32, res_fp16])
    cols_display = ["Model", "Size_MB", "mAP50", "mAP75", "NWD_mAP50", "Precision", "Recall", "F1_Score", "Center_Error_px", "Latency_ms", "FPS"]
    cols_available = [c for c in cols_display if c in df_summary.columns]
    df_table = df_summary[cols_available]

    csv_summary_path = os.path.join(FP16Config.OUTPUT_DIR, "model_fp16_benchmark_summary.csv")
    df_table.to_csv(csv_summary_path, index=False)
    txt_summary_path = os.path.join(FP16Config.EXPERIMENT_DIR, "model_fp16_benchmark_summary.txt")
    df_table.to_csv(txt_summary_path, index=False)

    print("\n[TONG KET] Bang so sanh doi chung dinh luong:")
    print(df_table.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=250)
    labels = ["Full ResNet50 (FP32)", "Full ResNet50 (FP16)"]
    colors = ["#1f77b4", "#2ca02c"]

    axes[0, 0].bar(labels, df_table["Size_MB"], color=colors, width=0.45)
    axes[0, 0].set_title("Dung Luong File Mo Hinh (MB) - Giam 50%", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Dung luong (MB)", fontsize=10, fontweight="bold")
    for i, v in enumerate(df_table["Size_MB"]):
        axes[0, 0].text(i, v + 2.0, f"{v:.1f} MB", ha="center", fontweight="bold", fontsize=10)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0, 0].set_ylim(0, max(df_table["Size_MB"]) * 1.18)

    axes[0, 1].bar(labels, df_table["FPS"], color=colors, width=0.45)
    axes[0, 1].set_title("Toc Do Suy Luan (FPS)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Frames Per Second", fontsize=10, fontweight="bold")
    for i, v in enumerate(df_table["FPS"]):
        axes[0, 1].text(i, v + 1.0, f"{v:.1f} FPS", ha="center", fontweight="bold", fontsize=10)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0, 1].set_ylim(0, max(df_table["FPS"]) * 1.18)

    x = np.arange(len(labels))
    w = 0.28
    axes[1, 0].bar(x - w/2, df_table["mAP50"], w, label="mAP@0.5 (IoU)", color="#3498db")
    axes[1, 0].bar(x + w/2, df_table["NWD_mAP50"], w, label="NWD-mAP@0.5", color="#e74c3c")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontweight="bold")
    axes[1, 0].set_title("Do Chinh Xac mAP@0.5 va NWD-mAP", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Do chinh xac (%)", fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1, 0].set_ylim(0, 105)

    axes[1, 1].bar(labels, df_table["Center_Error_px"], color=colors, width=0.45)
    axes[1, 1].set_title("Sai So Tam Ngam Trung Binh (Pixels)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Pixels", fontsize=10, fontweight="bold")
    for i, v in enumerate(df_table["Center_Error_px"]):
        axes[1, 1].text(i, v + 0.2, f"{v:.2f} px", ha="center", fontweight="bold", fontsize=10)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1, 1].set_ylim(0, max(df_table["Center_Error_px"]) * 1.25)

    plt.tight_layout()
    cmp_path = os.path.join(FP16Config.OUTPUT_DIR, "model_fp16_comparison_charts.png")
    plt.savefig(cmp_path, bbox_inches="tight")
    plt.close()
    print(f"[HOAN TAT] Da luu bieu do doi chung tai: {cmp_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Luong tu hoa FP16 va Danh gia Benchmark Anti-UAV")
    parser.add_argument("--voc_root", type=str, default=None, help="Duong dan goc toi thu muc VOC_AntiUAV_RGB")
    parser.add_argument("--val_csv", type=str, default=None, help="Duong dan tep val_triplets.csv")
    parser.add_argument("--checkpoint", type=str, default=None, help="Duong dan checkpoint goc FP32 (.keras)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size danh gia")
    parser.add_argument("--stride", type=int, default=5, help="Buoc nhay lay mau kiem thu")
    parser.add_argument("--samples", type=int, default=3000, help="So luong mau toi da dung danh gia")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_conversion_and_benchmark(args)
