# -*- coding: utf-8 -*-
"""
Evaluation Metrics: IoU, Normalized Wasserstein Distance (NWD), Precision, Recall, mAP
"""
import numpy as np

def compute_iou(b1, b2):
    """
    Computes Intersection over Union (IoU) between bounding box arrays.
    b1: [N, 4] or [4]
    b2: [N, 4] or [4]
    Boxes format: [xmin, ymin, xmax, ymax]
    """
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
    """
    Computes Normalized Gaussian Wasserstein Distance (NWD) for tiny objects:
    Wang et al., 'Normalized Gaussian Wasserstein Distance for Tiny Object Detection', arXiv 2021.
    """
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
    """
    Computes Average Precision (AP) using standard 101-point or continuous interpolation.
    """
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0, [0.0], [0.0]
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
