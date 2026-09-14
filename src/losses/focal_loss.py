# -*- coding: utf-8 -*-
"""
Modified Gaussian Focal Loss for CenterPoint Detection
"""
import tensorflow as tf

def gaussian_focal_loss(y_true, y_pred, alpha=2.0, beta=4.0):
    """
    Modified Gaussian Focal Loss for Heatmap Center Point Detection (Law & Deng, ECCV 2018):
    - y_true: [B, H, W, 1] ground truth Gaussian distribution (peak = 1.0, background = 0.0)
    - y_pred: [B, H, W, 1] predicted heatmap probabilities after sigmoid
    """
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
    neg_mask = tf.cast(tf.less(y_true, 1.0), tf.float32)

    pos_loss = -tf.pow(1.0 - y_pred, alpha) * tf.math.log(y_pred) * pos_mask
    neg_loss = -tf.pow(1.0 - y_true, beta) * tf.pow(y_pred, alpha) * tf.math.log(1.0 - y_pred) * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / tf.maximum(num_pos, 1.0)
    return loss
