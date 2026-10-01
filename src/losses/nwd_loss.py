import tensorflow as tf
# ham loss Normalized Gaussian Wasserstein Distance cho tiny object

def compute_nwd_size_loss(y_true_sz, y_pred_sz, mask, c_norm = 0.1):
    """
    cong thuc W_2^2 = ((w - w^)^2 + (h-h^)^2)/4, NWD = exp(-sqrt(W_2^2)/c_norm)
    l_size = 1/N sum sum (mask)(lamda|s_xy - s^2_xy|1 +  (1 - NWD(s_xy, s^_xy)) lambda = 5
    """

    w1, h1 = y_true_sz[..., 0], y_true_sz[..., 1]
    w2, h2 = y_pred_sz[..., 0], y_pred_sz[..., 1]

    num_pos = tf.reduce_sum(mask)
    size_dist_sq = (tf.square(w1 - w2) + tf.square(h1 - h2)) / 4.0
    w_dist = tf.sqrt(tf.maximum(size_dist_sq, 1e-7))
    nwd = tf.exp(-w_dist / c_norm)
    nwd_loss = (1 - nwd) * mask[..., 0]

    return tf.reduce_sum(nwd_loss) / tf.maximum(num_pos, 1e-7)