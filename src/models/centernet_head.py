# -*- coding: utf-8 -*-
"""
Decoupled Anchor-Free CenterNet Heads with Sigmoid Prior Size Head
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from .backbone import build_custom_resnet50v2_backbone
from .fpn import build_p2_fpn

class AnchorFreeCenterHead(layers.Layer):
    """
    Decoupled Anchor-Free Detection Head (CenterNet-style):
    - Branch 1: Heatmap (80, 80, 1) - Predicts probability of drone center point.
                Initialized with constant bias = -4.595 (prior prob = 0.01) to stabilize initial training.
    - Branch 2: Offset  (80, 80, 2) - Sub-pixel quantization error (dx, dy).
    - Branch 3: Size    (80, 80, 2) - Normalized width and height (w, h) in [0, 1].
                Uses Sigmoid activation with Logit Prior bias (-2.66) to eliminate Dying ReLU and ensure
                continuous non-zero gradient flow for micro-objects (< 32px).
    """
    def __init__(self, fpn_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.fpn_dim = fpn_dim

    def build(self, input_shape):
        # 1. Heatmap Branch
        self.hm_conv1 = layers.Conv2D(128, 3, padding="same", activation="relu", name="hm_conv1")
        self.hm_bn1 = layers.BatchNormalization(name="hm_bn1")
        self.hm_out = layers.Conv2D(
            1, 1, padding="same", activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-4.595),
            dtype="float32", name="heatmap"
        )

        # 2. Offset Branch
        self.off_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu", name="off_conv1")
        self.off_bn1 = layers.BatchNormalization(name="off_bn1")
        self.off_out = layers.Conv2D(2, 1, padding="same", dtype="float32", name="offset")

        # 3. Size Branch (w, h) - Sigmoid + Logit Prior bias (-2.66)
        self.size_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu", name="size_conv1")
        self.size_bn1 = layers.BatchNormalization(name="size_bn1")
        self.size_out = layers.Conv2D(
            2, 1, padding="same", activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.66),
            dtype="float32", name="size"
        )

        super().build(input_shape)

    def call(self, x):
        # Heatmap
        h_hm = self.hm_bn1(self.hm_conv1(x))
        pred_hm = self.hm_out(h_hm)

        # Offset
        h_off = self.off_bn1(self.off_conv1(x))
        pred_off = self.off_out(h_off)

        # Size
        h_size = self.size_bn1(self.size_conv1(x))
        pred_size = self.size_out(h_size)

        return {
            "heatmap": pred_hm,
            "offset": pred_off,
            "size": pred_size
        }

    def get_config(self):
        config = super().get_config()
        config.update({"fpn_dim": self.fpn_dim})
        return config

def build_detection_model(input_shape=(640, 640, 9), fpn_dim=128, name="ResNet50v2_P2FPN"):
    """
    Constructs the end-to-end Anchor-Free CenterNet model.
    Input: Temporal Triplet Tensor of shape (B, 640, 640, 9) representing (t-1, t, t+1) RGB frames.
    Output: Dictionary containing 'heatmap', 'offset', 'size' maps at 80x80 resolution (stride 8).
    """
    inputs = layers.Input(shape=input_shape, name="temporal_input")
    c2, c3, c4, c5 = build_custom_resnet50v2_backbone(inputs)
    p2, p3, p4, p5 = build_p2_fpn(c2, c3, c4, c5, fpn_dim=fpn_dim)

    # Multi-resolution fusion to unified stride 8
    p2_down = layers.MaxPooling2D(pool_size=2, name="fuse_p2_down")(p2)
    p3_same = p3
    p4_up   = layers.UpSampling2D(size=2, name="fuse_p4_up")(p4)
    p5_up   = layers.UpSampling2D(size=4, name="fuse_p5_up")(p5)

    fused = layers.Concatenate(axis=-1, name="fuse_concat")([p2_down, p3_same, p4_up, p5_up])
    fused = layers.BatchNormalization(name="fuse_bn")(
        layers.Conv2D(fpn_dim, kernel_size=3, padding="same", activation="relu", name="fuse_conv")(fused)
    )

    outputs = AnchorFreeCenterHead(fpn_dim=fpn_dim, name="anchor_free_head")(fused)
    return Model(inputs=inputs, outputs=outputs, name=name)
