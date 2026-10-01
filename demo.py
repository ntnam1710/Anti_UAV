# -*- coding: utf-8 -*-
"""
Kịch bản thực thi suy luận (Inference) cho mô hình CenterNet RGB phát hiện và bám bắt UAV.
Hỗ trợ:
- Tệp ảnh cục bộ hoặc đường dẫn URL ảnh từ xa
- Tệp video cục bộ hoặc đường dẫn URL video từ xa
- Luồng video trực tiếp từ Webcam
Tích hợp giao diện hiển thị tác chiến Tactical HUD với bộ lọc quỹ đạo quán tính Trajectory EMA Filter.
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Đảm bảo mã hóa UTF-8 cho luồng xuất chuẩn trên hệ điều hành Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import Config
from src.models import build_detection_model, load_trained_model, decode_detections
from src.tracking import TrajectoryEMAFilter, draw_hud_reticle, create_tactical_canvas
from src.utils import download_file_from_url

# Cấu hình cấp phát bộ nhớ GPU linh hoạt (Memory Growth)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def is_url(path):
    """Kiểm tra đường dẫn có phải là liên kết web (URL) hay không."""
    return path.startswith("http://") or path.startswith("https://")

def is_image_file(path):
    """Kiểm tra định dạng tệp ảnh hỗ trợ."""
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

def is_video_file(path):
    """Kiểm tra định dạng tệp video hỗ trợ."""
    ext = os.path.splitext(path)[1].lower()
    return ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".flv"]

def load_and_preprocess_frame(frame, target_size=(640, 640)):
    """Chuyển đổi không gian màu sang RGB, thay đổi kích thước và chuẩn hóa về [0.0, 1.0]."""
    im = cv2.resize(frame, target_size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / 255.0

def process_image(image_path, model, output_path, conf_thresh=0.40):
    """Xử lý suy luận trên một tệp ảnh đơn lẻ."""
    print(f"[THONG TIN] Dang xu ly anh: {image_path}")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Khong the doc tep anh tai: {image_path}")
    orig_h, orig_w = raw_img.shape[:2]

    # Tạo bộ ba giả lập (pseudo-triplet) từ khung hình tĩnh cho đầu vào 9 kênh (t-1, t, t+1)
    norm_frame = load_and_preprocess_frame(raw_img)
    tensor_input = np.concatenate([norm_frame, norm_frame, norm_frame], axis=-1)[np.newaxis, ...]

    # Thực hiện lan truyền xuôi qua mạng nơ-ron
    preds = model(tensor_input, training=False)
    score, bboxes = decode_detections(preds["heatmap"], preds["offset"], preds["size"])
    conf = float(score[0].numpy())
    box_norm = bboxes[0].numpy()
    abs_box = [box_norm[0] * orig_w, box_norm[1] * orig_h, box_norm[2] * orig_w, box_norm[3] * orig_h]

    disp_img = raw_img.copy()
    if conf >= conf_thresh:
        draw_hud_reticle(disp_img, abs_box, conf=conf, speed=0.0, state="DETECTED")
        print(f"[PHAT HIEN] Phat hien UAV! Do tin cay: {conf*100:.1f}%, BBox: [{abs_box[0]:.0f}, {abs_box[1]:.0f}, {abs_box[2]:.0f}, {abs_box[3]:.0f}]")
    else:
        print(f"[THONG TIN] Khong phat hien UAV vuot nguong tin cay {conf_thresh:.2f} (Diem cao nhat: {conf*100:.1f}%)")

    # Tạo giao diện bảng điều khiển tác chiến
    canvas = create_tactical_canvas(
        frame_v=disp_img,
        frame_aux=None,
        hud_title="ANTI-UAV RGB",
        state="DETECTED" if conf >= conf_thresh else "NO TARGET",
        frame_info=f"Source: {os.path.basename(image_path)} | Conf: {conf*100:.1f}%"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[HOAN TAT] Da luu anh chu thich tai: {output_path}")
    return output_path

def process_video(video_source, model, output_path, conf_thresh=0.40, max_frames=None, is_cam=False):
    """Xử lý suy luận trên chuỗi video hoặc luồng camera trực tiếp."""
    cap = cv2.VideoCapture(int(video_source) if is_cam else video_source)
    if not cap.isOpened():
        raise ValueError(f"Khong the mo nguon video: {video_source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 1000
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if max_frames:
        total_frames = min(total_frames, max_frames)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    out_w, out_h = 960, 540
    canvas_w = out_w
    canvas_h = out_h + 60

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    raw_avi = output_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw_avi, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h))

    tracker = TrajectoryEMAFilter(alpha=0.6, max_missing=15)

    # Bộ đệm trượt lưu 3 khung hình liên tiếp phục vụ Temporal Triplet
    frame_buf = []
    processed_count = 0

    print(f"[THONG TIN] Bat dau suy luan video: {video_source} ({total_frames} khung hinh)...")
    pbar = tqdm(total=total_frames, desc="Xu ly khung hinh")

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

        # Chuẩn bị tensor 9 kênh
        v_tm1 = load_and_preprocess_frame(f_tm1)
        v_t   = load_and_preprocess_frame(f_t)
        v_tp1 = load_and_preprocess_frame(f_tp1)
        tensor_in = np.concatenate([v_tm1, v_t, v_tp1], axis=-1)[np.newaxis, ...]

        # Suy luận qua CenterNet
        preds = model(tensor_in, training=False)
        score, bboxes = decode_detections(preds["heatmap"], preds["offset"], preds["size"])
        conf = float(score[0].numpy())
        box_n = bboxes[0].numpy()
        abs_box = [box_n[0] * orig_w, box_n[1] * orig_h, box_n[2] * orig_w, box_n[3] * orig_h]

        # Cập nhật trạng thái bộ lọc bám bắt quỹ đạo EMA
        final_box, state, spd = tracker.update(abs_box, conf >= conf_thresh)

        # Vẽ giao diện trên khung hình duy nhất Camera Visible RGB
        disp_v = f_t.copy()
        sx, sy = out_w / orig_w, out_h / orig_h
        disp_v_small = cv2.resize(disp_v, (out_w, out_h))
        if final_box is not None:
            scaled_box = [final_box[0] * sx, final_box[1] * sy, final_box[2] * sx, final_box[3] * sy]
            draw_hud_reticle(disp_v_small, scaled_box, conf=conf, speed=spd, state=state)

        # Bảng điều khiển tác chiến trên 1 Camera Visible duy nhất
        canvas = create_tactical_canvas(
            frame_v=disp_v_small,
            frame_aux=None,
            hud_title="ANTI-UAV RGB",
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

    # Chuyen doi ma hoa sang chuan H.264 tuong thich trinh duyet web qua ffmpeg neu co san
    import shutil
    has_ffmpeg = shutil.which("ffmpeg") is not None
    converted = False
    if has_ffmpeg:
        cmd = f'ffmpeg -y -i "{raw_avi}" -vcodec libx264 -pix_fmt yuv420p -crf 23 "{output_path}" -loglevel error'
        ret = os.system(cmd)
        if ret == 0 and os.path.exists(output_path):
            converted = True
            try:
                os.remove(raw_avi)
            except Exception:
                pass
            print(f"[HOAN TAT] Video ma hoa H.264 da tao thanh cong tai: {output_path}")

    if not converted:
        if os.path.exists(raw_avi):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(raw_avi, output_path)
        print(f"[HOAN TAT] Video da tao thanh cong tai: {output_path}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Chuong trinh suy luan Anti-UAV CenterNet RGB")
    parser.add_argument("--source", type=str, required=True,
                        help="Duong dan anh, video, hoac URL hoac chi so webcam ('0')")
    parser.add_argument("--checkpoint", type=str, default=Config.DEFAULT_CHECKPOINT,
                        help="Duong dan den tep trong so mo hinh .keras")
    parser.add_argument("--output", type=str, default=None,
                        help="Duong dan tep dau ra cho anh hoac video chu thich")
    parser.add_argument("--conf_thresh", type=float, default=0.40,
                        help="Nguong tin cay de loc hop bao phat hien (mac dinh: 0.40).")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="So luong khung hinh toi da can xu ly")
    parser.add_argument("--webcam", action="store_true",
                        help="Chay suy luan truc tiep qua webcam")

    args = parser.parse_args()

    # Tu dong tai xuong neu la URL
    source = args.source
    if is_url(source):
        source = download_file_from_url(source)

    # Kiem tra trong so mo hinh
    if not os.path.exists(args.checkpoint):
        print(f"[CANH BAO] Khong tim thay checkpoint tai: {args.checkpoint}")
        print("[INFO] Dang tim kiem cac tep trong so .keras thay the...")
        import glob
        cands = glob.glob("**/*.keras", recursive=True)
        if cands:
            args.checkpoint = cands[0]
            print(f"[INFO] Phat hien checkpoint thay the: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Khong tim thay checkpoint. Tai ve tu: {Config.KAGGLE_CHECKPOINT_URL}")

    # Khoi tao va nap trong so mo hinh
    print(f"[INFO] Dang nap mo hinh tu: {args.checkpoint}...")
    model = load_trained_model(args.checkpoint, input_shape=Config.INPUT_SHAPE, fpn_dim=Config.FPN_DIM)
    print("[HOAN TAT] Mo hinh da duoc khoi tao thanh cong!")

    # Duong dan xuat mac dinh
    if args.output is None:
        os.makedirs("outputs", exist_ok=True)
        if is_image_file(source):
            args.output = os.path.join("outputs", "annotated_" + os.path.basename(source))
        else:
            args.output = os.path.join("outputs", "annotated_" + os.path.splitext(os.path.basename(source))[0] + ".mp4")

    # Thuc thi suy luan
    if args.webcam or source.isdigit():
        process_video(source, model, args.output, conf_thresh=args.conf_thresh, max_frames=args.max_frames, is_cam=True)
    elif is_image_file(source):
        process_image(source, model, args.output, conf_thresh=args.conf_thresh)
    else:
        process_video(source, model, args.output, conf_thresh=args.conf_thresh, max_frames=args.max_frames, is_cam=False)

if __name__ == "__main__":
    main()
