#danh gia
import os
import sys
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import Config
from src.models import build_detection_model, load_trained_model, decode_detections
from src.utils.metrics import compute_iou, compute_nwd, compute_ap

#cau hinh gpu
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def decode_and_resize_eval(file_path, channels=3, target_size=(640, 640)):
    """Đọc ảnh từ đĩa, thay đổi kích thước và chuẩn hóa phục vụ đánh giá."""
    img_raw = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img_raw, channels=channels, ratio=2)
    img = tf.image.resize(img, target_size, method="bilinear")
    return tf.cast(img, tf.float32) / 255.0

def create_eval_dataset(csv_path, voc_root, batch_size=8, target_size=(640, 640), stride=1, max_samples=None):
    """Xây dựng tf.data.Dataset cho tập kiểm thử với hỗ trợ lấy mẫu theo chu kỳ (stride) và giới hạn số lượng mẫu."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Khong tim thay tep CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    orig_total = len(df)

    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    if max_samples and max_samples > 0 and len(df) > max_samples:
        indices = np.linspace(0, len(df) - 1, max_samples, dtype=int)
        df = df.iloc[indices].reset_index(drop=True)

    total_samples = len(df)
    print(f"   + Tong so mau co san: {orig_total} | Tap danh gia: {total_samples} mau (Stride: {stride}) | Batch: {batch_size}")

    orig_w, orig_h = 1920.0, 1080.0

    def preprocess_eval(vis_tm1_rel, vis_t_rel, vis_tp1_rel,
                        xmin, ymin, xmax, ymax, exist):
        p_vis_tm1 = tf.strings.join([voc_root, "/", vis_tm1_rel])
        p_vis_t   = tf.strings.join([voc_root, "/", vis_t_rel])
        p_vis_tp1 = tf.strings.join([voc_root, "/", vis_tp1_rel])

        img_v_tm1 = decode_and_resize_eval(p_vis_tm1, 3, target_size)
        img_v_t   = decode_and_resize_eval(p_vis_t,   3, target_size)
        img_v_tp1 = decode_and_resize_eval(p_vis_tp1, 3, target_size)

        tensor_input = tf.concat([img_v_tm1, img_v_t, img_v_tp1], axis=-1)

        norm_xmin = tf.clip_by_value(tf.cast(xmin, tf.float32) / orig_w, 0.0, 1.0)
        norm_ymin = tf.clip_by_value(tf.cast(ymin, tf.float32) / orig_h, 0.0, 1.0)
        norm_xmax = tf.clip_by_value(tf.cast(xmax, tf.float32) / orig_w, 0.0, 1.0)
        norm_ymax = tf.clip_by_value(tf.cast(ymax, tf.float32) / orig_h, 0.0, 1.0)

        norm_bbox = tf.where(
            exist == 1,
            tf.stack([norm_xmin, norm_ymin, norm_xmax, norm_ymax]),
            tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        )

        labels = {
            "class_output": tf.cast(exist, tf.float32),
            "bbox_output": tf.cast(norm_bbox, tf.float32)
        }
        return tensor_input, labels

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

    ds = ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, total_samples

def evaluate(voc_root, csv_path, checkpoint_path, output_dir="assets",
             batch_size=8, conf_thresh=0.40, stride=5, max_samples=3000):
    """Thực hiện toàn bộ quy trình kiểm thử và vẽ đồ thị Precision-Recall Curve."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" BAT DAU DANH GIA HIEN NANG DINH LUONG (RGB-ONLY)")
    print(f" Trong so       : {checkpoint_path}")
    print(f" CSV kiem thu   : {csv_path}")
    print(f" Kich thuoc lo  : {batch_size}")
    print(f" Nguong tin cay : {conf_thresh}")
    print("=" * 70)

    model = load_trained_model(checkpoint_path, input_shape=Config.INPUT_SHAPE, fpn_dim=Config.FPN_DIM)

    eval_ds, n_eval = create_eval_dataset(
        csv_path, voc_root, batch_size=batch_size,
        stride=stride, max_samples=max_samples
    )

    y_true_cls = []
    y_scores = []
    matches_iou50 = []
    matches_iou75 = []
    matches_nwd50 = []
    center_errors_px = []

    total_inference_time = 0.0
    total_frames = 0

    @tf.function
    def predict_step(x):
        preds = model(x, training=False)
        return decode_detections(preds["heatmap"], preds["offset"], preds["size"])

    print("\n[INFO] Dang thuc hien suy luan tren tap kiem thu...")
    for x_b, y_b in tqdm(eval_ds, total=int(np.ceil(n_eval / batch_size)), desc="Kiem thu"):
        t_start = time.time()
        scores, pred_bboxes = predict_step(x_b)
        elapsed = time.time() - t_start

        total_inference_time += elapsed
        total_frames += x_b.shape[0]

        b_exist = y_b["class_output"].numpy().astype(bool)
        b_gt_boxes = y_b["bbox_output"].numpy()
        b_pred_scores = scores.numpy()
        b_pred_boxes = pred_bboxes.numpy()

        for gt_exist, gt_box, score, pred_box in zip(b_exist, b_gt_boxes, b_pred_scores, b_pred_boxes):
            y_true_cls.append(gt_exist)
            y_scores.append(score)

            if gt_exist:
                iou = float(compute_iou(gt_box, pred_box))
                nwd = float(compute_nwd(gt_box, pred_box))

                cx_gt, cy_gt = (gt_box[0] + gt_box[2]) / 2.0 * 1920.0, (gt_box[1] + gt_box[3]) / 2.0 * 1080.0
                cx_pr, cy_pr = (pred_box[0] + pred_box[2]) / 2.0 * 1920.0, (pred_box[1] + pred_box[3]) / 2.0 * 1080.0
                c_err = np.sqrt((cx_gt - cx_pr)**2 + (cy_gt - cy_pr)**2)

                matches_iou50.append(iou >= 0.50)
                matches_iou75.append(iou >= 0.75)
                matches_nwd50.append(nwd >= 0.50)
                if score >= conf_thresh:
                    center_errors_px.append(c_err)
            else:
                matches_iou50.append(False)
                matches_iou75.append(False)
                matches_nwd50.append(False)

    y_true_cls = np.array(y_true_cls)
    y_scores = np.array(y_scores)
    matches_iou50 = np.array(matches_iou50)
    matches_iou75 = np.array(matches_iou75)
    matches_nwd50 = np.array(matches_nwd50)

    fps = total_frames / max(1e-5, total_inference_time)

    # Tính toán các chỉ số thống kê
    ap50, rec_arr, prec_arr = compute_ap(y_true_cls, y_scores, matches_iou50)
    ap75, _, _ = compute_ap(y_true_cls, y_scores, matches_iou75)
    nwd_ap, _, _ = compute_ap(y_true_cls, y_scores, matches_nwd50)

    pos_pred = y_scores >= conf_thresh
    tp = np.sum(pos_pred & y_true_cls & matches_iou50)
    fp = np.sum(pos_pred & ((~y_true_cls) | (~matches_iou50)))
    fn = np.sum((~pos_pred) & y_true_cls)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-7, precision + recall)
    mean_err_px = np.mean(center_errors_px) if len(center_errors_px) > 0 else 0.0

    print("\n" + "=" * 70)
    print(" KET QUA DANH GIA BENCHMARK ANTI-UAV")
    print("=" * 70)
    print(f" Do chinh xac (Precision IoU >= 0.5) : {precision * 100:.2f} %")
    print(f" Do thu hoi (Recall IoU >= 0.5)      : {recall * 100:.2f} %")
    print(f" Diem F1-Score                       : {f1 * 100:.2f} %")
    print(f" mAP@0.5                             : {ap50 * 100:.2f} %")
    print(f" mAP@0.75                            : {ap75 * 100:.2f} %")
    print(f" NWD-mAP@0.5 (Doi tuong sieu nho)    : {nwd_ap * 100:.2f} %")
    print(f" Sai so tam trung binh (Center Error): {mean_err_px:.1f} pixels")
    print(f" Toc do suy luan (Inference Speed)   : {fps:.1f} FPS")
    print("=" * 70)

    # Vẽ và lưu đồ thị đường cong Precision-Recall
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(rec_arr, prec_arr, color="#1f77b4", linewidth=2.5, label=f"CenterNet RGB (mAP@0.5 = {ap50*100:.1f}%)")
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Duong cong Precision-Recall (Tap kiem thu Anti-UAV)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"[HOAN TAT] Do thi PR Curve da duoc luu tai: {pr_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danh gia dinh luong CenterNet RGB")
    parser.add_argument("--voc_root", type=str, required=True, help="Thu muc goc dataset VOC")
    parser.add_argument("--csv_path", type=str, required=True, help="Tep CSV kiem thu")
    parser.add_argument("--checkpoint", type=str, default=Config.DEFAULT_CHECKPOINT, help="Duong dan checkpoint")
    parser.add_argument("--output_dir", type=str, default="assets", help="Thu muc xuat ket qua")
    parser.add_argument("--batch_size", type=int, default=8, help="Kich thuoc lo")
    parser.add_argument("--conf_thresh", type=float, default=0.40, help="Nguong tin cay")
    parser.add_argument("--stride", type=int, default=5, help="Buoc lay mau khung hinh")
    parser.add_argument("--max_samples", type=int, default=3000, help="So mau kiem thu toi da")
    args = parser.parse_args()

    evaluate(
        voc_root=args.voc_root,
        csv_path=args.csv_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        conf_thresh=args.conf_thresh,
        stride=args.stride,
        max_samples=args.max_samples
    )
