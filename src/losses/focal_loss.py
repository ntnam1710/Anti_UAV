#ham gaussian focal loss de tim tam cua object

import tensorflow as tf

def gaussian_focal_loss(y_true, y_pred, alpha = 2.0, beta = 4.0):
    """
    Cong thuc: if y_true == 1: -1/N sum(tu x = 1 den W/R) sum(tu y = 1 den H/R) (1-y_pred)^a * log(y_pred)
             else: -1/N sum(tu x = 1 den W/R) sum(tu y = 1 den H/R) (1-y_true)^b * (y_pred)^a * log(1-y_pred)
        N: tong so pixel trong heatmap
        a,b: he so trong de dieu chinh trong so cua cac sample
        y_true: heatmap gaussian that 
        y_pred: heatmap gaussian du doan
        alpha = 2, beta = 4
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