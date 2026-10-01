#ham masked sub-pixel offset l1 loss de hoi quy size oject va do lech

import tensorflow as tf

def masked_l1_loss(y_true, y_pred, mask):
    #cong thuc 1/N sum |y_true - y_pred| * mask
    num_pos = tf.reduce_sum(mask)
    diff = tf.abs(y_true - y_pred) * mask
    loss = tf.reduce_sum(diff) / tf.maximum(num_pos, 1.0)
    return loss