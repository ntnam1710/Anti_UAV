# -*- coding: utf-8 -*-
"""
Normalized Gaussian Wasserstein Distance (NWD) Loss for Tiny Object Detection
"""
import tensorflow as tf

def compute_nwd_size_loss(y_true_sz, y_pred_sz, mask, c_norm=0.10):
    """
    Computes scale-invariant Gaussian Wasserstein Distance loss between predicted and GT bounding box sizes:
    W_2^2(N_a, N_b) = ||m_a - m_b||_2^2 + Tr(Sigma_a + Sigma_b - 2(Sigma_a^{1/2} Sigma_b Sigma_a^{1/2})^{1/2})
    For aligned bounding boxes: (w1 - w2)^2 / 4 + (h1 - h2)^2 / 4
    NWD = exp(-sqrt(dist_sq) / C)
    """
    w1, h1 = y_true_sz[..., 0], y_true_sz[..., 1]
    w2, h2 = y_pred_sz[..., 0], y_pred_sz[..., 1]
    size_dist_sq = (tf.square(w1 - w2) + tf.square(h1 - h2)) / 4.0
    w_dist = tf.sqrt(tf.maximum(size_dist_sq, 1e-7))
    nwd = tf.exp(-w_dist / c_norm)
    nwd_loss = (1.0 - nwd) * mask[..., 0]
    num_pos = tf.reduce_sum(mask)
    return tf.reduce_sum(nwd_loss) / tf.maximum(num_pos, 1.0)
