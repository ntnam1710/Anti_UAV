# -*- coding: utf-8 -*-
"""
Các độ đo đánh giá định lượng: IoU, Normalized Gaussian Wasserstein Distance (NWD), Precision, Recall, mAP.
"""
import numpy as np

def compute_iou(b1, b2):
    r"""
    Tính toán chỉ số Intersection over Union (IoU) giữa hai tập hợp Bounding Box:
    \[
    \text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}
    \]
    
    Tham số:
        b1: Mảng tọa độ [N, 4] hoặc [4] theo định dạng [xmin, ymin, xmax, ymax].
        b2: Mảng tọa độ [N, 4] hoặc [4] theo định dạng [xmin, ymin, xmax, ymax].
        
    Trả về:
        np.ndarray hoặc float: Giá trị IoU trong đoạn [0.0, 1.0].
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
    r"""
    Tính toán độ tương đồng Normalized Gaussian Wasserstein Distance (NWD) chuyên dụng cho đối tượng siêu nhỏ:
    Tham khảo: Wang et al., 'Normalized Gaussian Wasserstein Distance for Tiny Object Detection', arXiv 2021.
    
    Công thức:
    \[
    \text{NWD}(N_a, N_b) = \exp\left(-\frac{\sqrt{W_2^2(N_a, N_b)}}{C}\right)
    \]
    
    Tham số:
        b1: Tọa độ [xmin, ymin, xmax, ymax] chuẩn hóa.
        b2: Tọa độ [xmin, ymin, xmax, ymax] chuẩn hóa.
        c_norm (float): Hằng số điều chỉnh độ nhạy khoảng cách (mặc định: 0.02).
        
    Trả về:
        np.ndarray hoặc float: Giá trị NWD trong đoạn [0.0, 1.0].
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
    Tính toán chỉ số Average Precision (AP) theo phương pháp nội suy đường cong Precision-Recall liên tục tiêu chuẩn VOC/COCO.
    
    Tham số:
        y_true: Mảng boolean/int đánh dấu đối tượng thực tế tồn tại.
        y_scores: Mảng điểm số tin cậy dự báo.
        matches: Mảng boolean đánh dấu dự báo khớp với ground truth (IoU/NWD vượt ngưỡng).
        
    Trả về:
        tuple: (ap, recall_curve, precision_curve)
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
