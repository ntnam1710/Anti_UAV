# -*- coding: utf-8 -*-
"""
Masked L1 Loss for Sub-Pixel Offset and Bounding Box Dimensions
"""
import tensorflow as tf

def masked_l1_loss(y_true, y_pred, mask):
    """
    Masked L1 regression loss evaluated strictly at positive center locations (mask == 1).
    """
    diff = tf.abs(y_true - y_pred) * mask
    num_pos = tf.reduce_sum(mask)
    return tf.reduce_sum(diff) / tf.maximum(num_pos, 1.0)
