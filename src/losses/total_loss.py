import tensorflow as tf
from .focal_loss import gaussian_focal_loss
from .l1_loss import masked_l1_loss
from .nwd_loss import compute_nwd_size_loss

def size_regression_loss(y_true_sz, y_pred_sz, mask, c_norm = 0.10):
    l1_loss = masked_l1_loss(y_true_sz, y_pred_sz, mask)
    nwd_loss = compute_nwd_size_loss(y_true_sz, y_pred_sz, mask, c_norm)
    return l1_loss * 5 + nwd_loss

def compute_combined_loss(y_true_dict, y_pred_dict, lambda_hm = 1.0, lambda_off = 1.0, lambda_size = 1.0):
    y_true_hm = tf.cast(y_true_dict["heatmap"], tf.float32)
    y_pred_hm = tf.cast(y_pred_dict["heatmap"], tf.float32)

    y_true_off = tf.cast(y_true_dict["offset"], tf.float32)
    y_pred_off = tf.cast(y_pred_dict["offset"], tf.float32)

    y_true_size = tf.cast(y_true_dict["size"], tf.float32)
    y_pred_size = tf.cast(y_pred_dict["size"], tf.float32)

    mask = tf.cast(y_true_dict["mask"], tf.float32)

    loss_hm = gaussian_focal_loss(y_true_hm, y_pred_hm)
    loss_off = masked_l1_loss(y_true_off, y_pred_off, mask)
    loss_size = size_regression_loss(y_true_size, y_pred_size, mask)
    
    total_loss = lambda_hm * loss_hm + lambda_off * loss_off + lambda_size * loss_size

    return total_loss, loss_hm, (lambda_off * loss_off + lambda_size * loss_size)