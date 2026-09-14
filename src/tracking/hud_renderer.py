# -*- coding: utf-8 -*-
"""
Military-Grade Tactical HUD (Heads-Up Display) Rendering Utilities
"""
import cv2
import numpy as np

def draw_hud_reticle(img, bbox, conf=1.0, speed=0.0, state="DETECTED"):
    """
    Renders tactical HUD crosshairs, corner brackets, and telemetry on the image.
    - DETECTED: Bright Green
    - TRACKED_EMA: Amber / Orange
    - LOST / Inactive: Dim Gray
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if state == "DETECTED":
        color = (0, 255, 0)
    elif state == "TRACKED_EMA":
        color = (0, 215, 255)
    else:
        color = (150, 150, 150)

    line_len = max(8, min(w, h) // 3)
    t = 2

    # Corner brackets
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, t)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, t)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, t)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, t)

    # Center crosshair dot
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.circle(img, (cx, cy), 2, color, -1)

    # Telemetry badge
    info_str = f"DRONE {conf*100:.1f}% | {w}x{h}px | v={speed:.1f}px/f"
    (fw, fh), _ = cv2.getTextSize(info_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    badge_y1 = max(0, y1 - fh - 8)
    badge_y2 = badge_y1 + fh + 6
    cv2.rectangle(img, (x1, badge_y1), (x1 + fw + 8, badge_y2), (20, 20, 20), -1)
    cv2.rectangle(img, (x1, badge_y1), (x1 + fw + 8, badge_y2), color, 1)
    cv2.putText(img, info_str, (x1 + 4, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def create_tactical_canvas(frame_v, frame_aux=None, hud_title="ANTI-UAV REALTIME TACTICAL TRACKING",
                            state="DETECTED", frame_info=""):
    """
    Creates a dual-view tactical console:
    Left: CAM 1 (Visible RGB Target Frame)
    Right: CAM 2 (Auxiliary Frame or Motion Heatmap)
    Header: Mission Telemetry HUD Banner
    """
    out_w, out_h = 960, 540
    canvas_w = out_w * 2 if frame_aux is not None else out_w
    canvas_h = out_h + 60

    disp_v = cv2.resize(frame_v, (out_w, out_h))
    cv2.putText(disp_v, "CAM 1: VISIBLE RGB (TARGET FRAME)", (15, out_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    if frame_aux is not None:
        disp_aux = cv2.resize(frame_aux, (out_w, out_h))
        cv2.putText(disp_aux, "CAM 2: TEMPORAL MOTION HEATMAP", (15, out_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        video_strip = np.hstack([disp_v, disp_aux])
    else:
        video_strip = disp_v

    # Header HUD Banner
    hud = np.zeros((60, canvas_w, 3), dtype=np.uint8)
    hud[:, :] = (20, 25, 30)
    cv2.putText(hud, hud_title, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if frame_info:
        cv2.putText(hud, frame_info, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    stat_color = (0, 255, 0) if state == "DETECTED" else ((0, 215, 255) if state == "TRACKED_EMA" else (100, 100, 100))
    cv2.putText(hud, f"STATUS: [{state}]", (canvas_w - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, stat_color, 2)

    return np.vstack([hud, video_strip])
