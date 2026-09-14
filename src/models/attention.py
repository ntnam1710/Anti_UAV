# -*- coding: utf-8 -*-
"""
Coordinate Attention Mechanism for Efficient Spatial-Channel Feature Recalibration
"""
import tensorflow as tf
from tensorflow.keras import layers

class CoordinateAttention(layers.Layer):
    """
    Coordinate Attention for Efficient Mobile Network Design (Hou et al., CVPR 2021).
    Encodes both channel relationships and positional information with precise coordinate orientation.
    """
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(8, channels // self.reduction)
        self.conv_shared = layers.Conv2D(reduced, kernel_size=1, strides=1, use_bias=True)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("relu")
        self.conv_h = layers.Conv2D(channels, kernel_size=1, activation="sigmoid", use_bias=True)
        self.conv_w = layers.Conv2D(channels, kernel_size=1, activation="sigmoid", use_bias=True)
        super().build(input_shape)

    def call(self, x):
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        
        # 1D Global Pooling along height and width
        x_h = tf.reduce_mean(x, axis=2, keepdims=True)  # (B, H, 1, C)
        x_w = tf.reduce_mean(x, axis=1, keepdims=True)  # (B, 1, W, C)
        x_w_perm = tf.transpose(x_w, perm=[0, 2, 1, 3]) # (B, W, 1, C)
        
        concat = tf.concat([x_h, x_w_perm], axis=1)     # (B, H+W, 1, C)
        y = self.act(self.bn(self.conv_shared(concat))) # (B, H+W, 1, C//r)
        
        y_h = y[:, :h, :, :]                            # (B, H, 1, C//r)
        y_w = tf.transpose(y[:, h:, :, :], perm=[0, 2, 1, 3]) # (B, 1, W, C//r)
        
        att_h = self.conv_h(y_h)                        # (B, H, 1, C)
        att_w = self.conv_w(y_w)                        # (B, 1, W, C)
        
        return x * att_h * att_w

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config
