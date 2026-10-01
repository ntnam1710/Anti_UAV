# -*- coding: utf-8 -*-
"""
renderer.py - Ket xuat do hoa giao dien tac chien HUD tren 1 Camera Visible duy nhat.
Tieu de hien thi: "ANTI-UAV RGB"
"""
import cv2
import numpy as np

def draw_reticle(img, bbox, conf=1.0, speed=0.0, state="DETECTED"):
    """
    Ve khung render va thong so muc tieu drone len anh.
    img (np.ndarray): Anh dau vao
    bbox (iterable): Toa do [xmin, ymin, xmax, ymax]
    conf (float): Do tin cay (0.0 - 1.0)
    speed (float): Toc do di chuyen (pixels/frame)
    state (str): Trang thai theo doi ("DETECTED", "OCCLUDED", "LOST")
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if state == "DETECTED":
        color = (0, 255, 0)
    elif state == "OCCLUDED":
        color = (0, 215, 255)
    else:
        color = (150, 150, 150)

    line_len = max(8, min(w, h) // 3)
    t = 2

    # Ve 4 goc reticle
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, t)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, t)

    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, t)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, t)

    # Khung box vien mong va tam doi tuong
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.circle(img, (cx, cy), 2, color, -1)

    # Bang thong so muc tieu
    infor_str = f"DRONE {conf*100:.1f}% | {w}x{h}px | v={speed:.1f}px/f"
    (fw, fh), _ = cv2.getTextSize(infor_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    badge_y1 = max(0, y1 - fh - 8)
    badge_y2 = badge_y1 + fh + 6
    cv2.rectangle(img, (x1, badge_y1), (x1 + fw + 8, badge_y2), (20, 20, 20), -1)
    cv2.rectangle(img, (x1, badge_y1), (x1 + fw + 8, badge_y2), color, 1)
    cv2.putText(img, infor_str, (x1 + 4, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def create_canvas(frame_v, frame_aux=None, title="ANTI-UAV RGB", hud_title=None, state="DETECTED", frame_info="", **kwargs):
    """
    Tao khung hinh tac chien tren 1 Camera Visible duy nhat voi thanh tieu de "ANTI-UAV RGB".
    frame_v (np.ndarray): Khung hinh camera Visible RGB chinh
    frame_aux: Khong su dung (da loai bo Cam 2 Motion Heatmap)
    title (str): Tieu de hien thi, mac dinh "ANTI-UAV RGB"
    state (str): Trang thai muc tieu
    frame_info (str): Thong tin phu ve khung hinh
    """
    if hud_title is not None:
        title = hud_title
    out_w, out_h = 960, 540
    canvas_w = out_w
    canvas_h = out_h + 60

    disp_v = cv2.resize(frame_v, (out_w, out_h))

    # Thanh tieu de tren cung (Header HUD)
    tieu_de = np.zeros((60, canvas_w, 3), dtype=np.uint8)
    tieu_de[:, :] = (20, 25, 30)

    # Tieu de chi ghi "ANTI-UAV RGB"
    main_title = "ANTI-UAV RGB" if not title or title.startswith("ANTI-UAV") else title
    cv2.putText(tieu_de, main_title, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if frame_info:
        cv2.putText(tieu_de, frame_info, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    stat_color = (0, 255, 0) if state == "DETECTED" else ((0, 215, 255) if state == "OCCLUDED" else (100, 100, 100))
    cv2.putText(tieu_de, f"STATUS: [{state}]", (canvas_w - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, stat_color, 2)

    return np.vstack([tieu_de, disp_v])

# Alias de tuong thich voi ca hai quy uoc ten goi
draw_hud_reticle = draw_reticle
create_tactical_canvas = create_canvas
