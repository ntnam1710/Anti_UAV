# -*- coding: utf-8 -*-
"""
Local Peak Decoding via 3x3 MaxPool (NMS-Free)
"""
import tensorflow as tf

def decode_detections(heatmap, offset, size, default_w=0.065, default_h=0.070, min_size=0.015):
    """
    Decodes predicted bounding boxes from Anchor-Free heatmaps using 3x3 MaxPool local peak detection:
    - Extracts peak center points via MaxPool2D(ksize=3, strides=1, padding='SAME')
    - Extracts Top-1 peak for Anti-UAV tracking (or single dominant drone target)
    - Fallback heuristic: If size prediction falls below min_size, injects Drone Prior (w=0.065, h=0.070)
    
    Returns:
    - score: Tensor of shape (B,) - Detection confidence score in [0, 1]
    - bboxes: Tensor of shape (B, 4) - Normalized [xmin, ymin, xmax, ymax] in [0, 1]
    """
    hmax = tf.nn.max_pool2d(heatmap, ksize=3, strides=1, padding="SAME")
    keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
    peak_heatmap = heatmap * keep

    B = tf.shape(heatmap)[0]
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2]

    flat_peaks = tf.reshape(peak_heatmap, [B, H * W])
    top_scores, top_indices = tf.math.top_k(flat_peaks, k=1)
    score = top_scores[:, 0]
    idx = top_indices[:, 0]

    grid_y = tf.cast(idx // W, tf.float32)
    grid_x = tf.cast(idx % W, tf.float32)

    flat_offset = tf.reshape(offset, [B, H * W, 2])
    flat_size = tf.reshape(size, [B, H * W, 2])

    batch_indices = tf.range(B, dtype=tf.int32)
    gather_idx = tf.stack([batch_indices, idx], axis=-1)

    peak_offset = tf.gather_nd(flat_offset, gather_idx)
    peak_size = tf.gather_nd(flat_size, gather_idx)

    # Reconstruct center coordinates on normalized grid [0, 1]
    cx = (grid_x + peak_offset[:, 0]) / tf.cast(W, tf.float32)
    cy = (grid_y + peak_offset[:, 1]) / tf.cast(H, tf.float32)
    bw = peak_size[:, 0]
    bh = peak_size[:, 1]

    # Prior fallback safeguard
    bw = tf.where(bw < min_size, tf.constant(default_w, dtype=bw.dtype), bw)
    bh = tf.where(bh < min_size, tf.constant(default_h, dtype=bh.dtype), bh)

    xmin = tf.clip_by_value(cx - bw / 2.0, 0.0, 1.0)
    ymin = tf.clip_by_value(cy - bh / 2.0, 0.0, 1.0)
    xmax = tf.clip_by_value(cx + bw / 2.0, 0.0, 1.0)
    ymax = tf.clip_by_value(cy + bh / 2.0, 0.0, 1.0)

    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return score, bboxes
