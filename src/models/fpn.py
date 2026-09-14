# -*- coding: utf-8 -*-
"""
P2-FPN Feature Pyramid Network with Coordinate Attention
"""
import tensorflow as tf
from tensorflow.keras import layers
from .attention import CoordinateAttention

def build_p2_fpn(c2, c3, c4, c5, fpn_dim=128):
    """
    Constructs top-down Feature Pyramid Network extending down to high-resolution P2 level (stride 4)
    with Coordinate Attention modules at P2 and P3 levels for micro-UAV enhancement.
    """
    lat_c5 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c5")(c5)
    lat_c4 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c4")(c4)
    lat_c3 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c3")(c3)
    lat_c2 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c2")(c2)

    p5 = lat_c5
    p4 = layers.Add(name="fpn_add_p4")([lat_c4, layers.UpSampling2D(size=2, name="fpn_up_p5")(p5)])
    p3 = layers.Add(name="fpn_add_p3")([lat_c3, layers.UpSampling2D(size=2, name="fpn_up_p4")(p4)])
    p2 = layers.Add(name="fpn_add_p2")([lat_c2, layers.UpSampling2D(size=2, name="fpn_up_p3")(p3)])

    p5 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p5")(p5)
    p4 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p4")(p4)
    p3 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p3")(p3)
    p2 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p2")(p2)

    # Apply Coordinate Attention to P2 and P3 to focus on fine-grained drone kinematics
    p2 = CoordinateAttention(name="ca_p2")(p2)
    p3 = CoordinateAttention(name="ca_p3")(p3)
    
    return p2, p3, p4, p5
