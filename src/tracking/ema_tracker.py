# -*- coding: utf-8 -*-
"""
Trajectory Exponential Moving Average (EMA) Filter for Temporal Stability
"""
import numpy as np

class TrajectoryEMAFilter:
    """
    Trajectory Exponential Moving Average (EMA) Filter:
    - Smooths bounding box jitter across consecutive video frames
    - Predicts inertial trajectory during brief occlusions or missed detections (up to max_missing frames)
    - Estimates instantaneous velocity in pixels/frame
    """
    def __init__(self, alpha=0.6, max_missing=15):
        self.alpha = alpha
        self.max_missing = max_missing
        self.smooth_box = None
        self.velocity = np.array([0.0, 0.0])
        self.missing_count = 0
        self.is_active = False

    def update(self, detected_box, is_detected):
        """
        Updates filter state with detected bounding box [xmin, ymin, xmax, ymax] or missing status.
        Returns:
            smooth_box: np.ndarray [4] or None
            state: "DETECTED" | "TRACKED_EMA" | "LOST"
            speed: float velocity magnitude in pixels/frame
        """
        if is_detected and detected_box is not None:
            cur_box = np.array(detected_box, dtype=np.float32)
            cur_cx = (cur_box[0] + cur_box[2]) / 2.0
            cur_cy = (cur_box[1] + cur_box[3]) / 2.0

            if self.smooth_box is None:
                self.smooth_box = cur_box
                self.velocity = np.array([0.0, 0.0])
            else:
                prev_cx = (self.smooth_box[0] + self.smooth_box[2]) / 2.0
                prev_cy = (self.smooth_box[1] + self.smooth_box[3]) / 2.0
                self.velocity = 0.6 * self.velocity + 0.4 * np.array([cur_cx - prev_cx, cur_cy - prev_cy])
                self.smooth_box = self.alpha * cur_box + (1.0 - self.alpha) * self.smooth_box

            self.missing_count = 0
            self.is_active = True
            return self.smooth_box.copy(), "DETECTED", float(np.linalg.norm(self.velocity))
        else:
            if self.is_active and self.missing_count < self.max_missing:
                self.missing_count += 1
                self.smooth_box[0] += self.velocity[0]
                self.smooth_box[2] += self.velocity[0]
                self.smooth_box[1] += self.velocity[1]
                self.smooth_box[3] += self.velocity[1]
                return self.smooth_box.copy(), "TRACKED_EMA", float(np.linalg.norm(self.velocity))
            else:
                self.is_active = False
                self.smooth_box = None
                return None, "LOST", 0.0
