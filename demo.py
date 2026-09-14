# -*- coding: utf-8 -*-
"""
Interactive & CLI Inference Script for Anti-UAV CenterNet RGB
Supports:
- Local Image file or Remote Image URL
- Local Video file or Remote Video URL
- Live Webcam stream
Renders tactical HUD overlay with Trajectory EMA Filter and outputs annotated media.
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Ensure UTF-8 stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import Config
from src.models import build_detection_model, decode_detections
from src.tracking import TrajectoryEMAFilter, draw_hud_reticle, create_tactical_canvas
from src.utils import download_file_from_url

# Configure GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def is_url(path):
    return path.startswith("http://") or path.startswith("https://")

def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

def is_video_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".flv"]

def load_and_preprocess_frame(frame, target_size=(640, 640)):
    im = cv2.resize(frame, target_size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / 255.0

def process_image(image_path, model, output_path, conf_thresh=0.20):
    print(f">> Processing single image: {image_path}")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    orig_h, orig_w = raw_img.shape[:2]

    # Construct pseudo-triplet from static image (t-1, t, t+1)
    norm_frame = load_and_preprocess_frame(raw_img)
    tensor_input = np.concatenate([norm_frame, norm_frame, norm_frame], axis=-1)[np.newaxis, ...]

    # Inference
    preds = model(tensor_input, training=False)
    score, bboxes = decode_detections(preds["heatmap"], preds["offset"], preds["size"])
    conf = float(score[0].numpy())
    box_norm = bboxes[0].numpy()
    abs_box = [box_norm[0] * orig_w, box_norm[1] * orig_h, box_norm[2] * orig_w, box_norm[3] * orig_h]

    # Draw Tactical Reticle
    disp_img = raw_img.copy()
    if conf >= conf_thresh:
        draw_hud_reticle(disp_img, abs_box, conf=conf, speed=0.0, state="DETECTED")
        print(f"🎯 Drone Detected! Conf: {conf*100:.1f}%, BBox: [{abs_box[0]:.0f}, {abs_box[1]:.0f}, {abs_box[2]:.0f}, {abs_box[3]:.0f}]")
    else:
        print(f"ℹ️ No drone detected above confidence threshold {conf_thresh:.2f} (Top score: {conf*100:.1f}%)")

    # Create Console Canvas
    canvas = create_tactical_canvas(
        frame_v=disp_img,
        frame_aux=None,
        hud_title="ANTI-UAV RGB CENTERNET INFERENCE",
        state="DETECTED" if conf >= conf_thresh else "NO TARGET",
        frame_info=f"Source: {os.path.basename(image_path)} | Conf: {conf*100:.1f}%"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"✅ Annotated image saved to: {output_path}")
    return output_path

def process_video(video_source, model, output_path, conf_thresh=0.15, max_frames=None, is_cam=False):
    cap = cv2.VideoCapture(int(video_source) if is_cam else video_source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {video_source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 1000
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if max_frames:
        total_frames = min(total_frames, max_frames)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    out_w, out_h = 960, 540
    canvas_w = out_w * 2
    canvas_h = out_h + 60

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    raw_avi = output_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw_avi, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h))

    tracker = TrajectoryEMAFilter(alpha=0.6, max_missing=15)

    # Frame sliding buffer for temporal triplet
    frame_buf = []
    processed_count = 0

    print(f">> Starting Video Inference: {video_source} ({total_frames} frames)...")
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buf.append(frame)
        if len(frame_buf) > 3:
            frame_buf.pop(0)

        if len(frame_buf) == 1:
            f_tm1, f_t, f_tp1 = frame_buf[0], frame_buf[0], frame_buf[0]
        elif len(frame_buf) == 2:
            f_tm1, f_t, f_tp1 = frame_buf[0], frame_buf[1], frame_buf[1]
        else:
            f_tm1, f_t, f_tp1 = frame_buf[0], frame_buf[1], frame_buf[2]

        # 9-channel tensor
        v_tm1 = load_and_preprocess_frame(f_tm1)
        v_t   = load_and_preprocess_frame(f_t)
        v_tp1 = load_and_preprocess_frame(f_tp1)
        tensor_in = np.concatenate([v_tm1, v_t, v_tp1], axis=-1)[np.newaxis, ...]

        # CenterNet forward
        preds = model(tensor_in, training=False)
        score, bboxes = decode_detections(preds["heatmap"], preds["offset"], preds["size"])
        conf = float(score[0].numpy())
        box_n = bboxes[0].numpy()
        abs_box = [box_n[0] * orig_w, box_n[1] * orig_h, box_n[2] * orig_w, box_n[3] * orig_h]

        # EMA Tracking update
        final_box, state, spd = tracker.update(abs_box, conf >= conf_thresh)

        # Draw on main frame
        disp_v = f_t.copy()
        sx, sy = out_w / orig_w, out_h / orig_h
        if final_box is not None:
            scaled_box = [final_box[0] * sx, final_box[1] * sy, final_box[2] * sx, final_box[3] * sy]
            disp_v_small = cv2.resize(disp_v, (out_w, out_h))
            draw_hud_reticle(disp_v_small, scaled_box, conf=conf, speed=spd, state=state)
        else:
            disp_v_small = cv2.resize(disp_v, (out_w, out_h))

        # Auxiliary CAM 2: Motion Difference Heatmap |I_t - I_{t-1}|
        diff = cv2.absdiff(f_t, f_tm1)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        aux_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        disp_aux_small = cv2.resize(aux_heatmap, (out_w, out_h))
        if final_box is not None:
            draw_hud_reticle(disp_aux_small, scaled_box, conf=conf, speed=spd, state=state)

        # Tactical Canvas
        canvas = create_tactical_canvas(
            frame_v=disp_v_small,
            frame_aux=disp_aux_small,
            hud_title="ANTI-UAV REALTIME TACTICAL TRACKING",
            state=state,
            frame_info=f"Frame: {processed_count+1:04d}/{total_frames} | FPS: {fps:.1f} | Drone: {spd:.1f} px/f"
        )

        writer.write(canvas)
        processed_count += 1
        pbar.update(1)

        if max_frames and processed_count >= max_frames:
            break

    pbar.close()
    cap.release()
    writer.release()

    # Transcode to web-compatible H.264 using ffmpeg if available
    cmd = f'ffmpeg -y -i "{raw_avi}" -vcodec libx264 -pix_fmt yuv420p -crf 23 "{output_path}" -loglevel error'
    ret = os.system(cmd)
    if ret == 0 and os.path.exists(output_path):
        try:
            os.remove(raw_avi)
        except Exception:
            pass
        print(f"✅ H.264 Encoded video successfully created: {output_path}")
    else:
        if os.path.exists(raw_avi):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(raw_avi, output_path)
        print(f"✅ Video successfully created: {output_path}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Anti-UAV CenterNet RGB Inference Demo")
    parser.add_argument("--source", type=str, required=True,
                        help="Path or URL to an image, video file, or camera index ('0')")
    parser.add_argument("--checkpoint", type=str, default=Config.DEFAULT_CHECKPOINT,
                        help="Path to pretrained .keras model weights")
    parser.add_argument("--output", type=str, default=None,
                        help="Destination output path for annotated image or video")
    parser.add_argument("--conf_thresh", type=float, default=0.15,
                        help="Confidence detection threshold")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process for video")
    parser.add_argument("--webcam", action="store_true",
                        help="Run live inference on webcam")

    args = parser.parse_args()

    # Auto download if URL
    source = args.source
    if is_url(source):
        source = download_file_from_url(source)

    # Validate Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"[!] Warning: Checkpoint not found at: {args.checkpoint}")
        print(">> Searching for alternative .keras weights in project...")
        import glob
        cands = glob.glob("**/*.keras", recursive=True)
        if cands:
            args.checkpoint = cands[0]
            print(f">> Found checkpoint: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint not found. Download from: {Config.KAGGLE_CHECKPOINT_URL}")

    # Build Model and Load Weights
    print(f">> Loading model architecture (Input: {Config.INPUT_SHAPE})...")
    model = build_detection_model(input_shape=Config.INPUT_SHAPE, fpn_dim=Config.FPN_DIM)
    print(f">> Loading pretrained weights from: {args.checkpoint}...")
    model.load_weights(args.checkpoint)
    print("✅ Model successfully initialized!")

    # Default output path
    if args.output is None:
        os.makedirs("outputs", exist_ok=True)
        if is_image_file(source):
            args.output = os.path.join("outputs", "annotated_" + os.path.basename(source))
        else:
            args.output = os.path.join("outputs", "annotated_" + os.path.splitext(os.path.basename(source))[0] + ".mp4")

    # Run inference
    if args.webcam or source.isdigit():
        process_video(source, model, args.output, conf_thresh=args.conf_thresh, max_frames=args.max_frames, is_cam=True)
    elif is_image_file(source):
        process_image(source, model, args.output, conf_thresh=args.conf_thresh)
    else:
        process_video(source, model, args.output, conf_thresh=args.conf_thresh, max_frames=args.max_frames, is_cam=False)

if __name__ == "__main__":
    main()
