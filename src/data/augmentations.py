# -*- coding: utf-8 -*-
"""
Data Augmentation and Image Preprocessing Utilities
"""
import cv2
import numpy as np
import tensorflow as tf

def decode_and_resize_img(file_path, channels=3, target_size=(640, 640), ratio=1):
    """
    Decodes JPEG image and resizes to target shape.
    Supports ratio downsampling for accelerated decoding.
    """
    img_raw = tf.io.read_file(file_path)
    if ratio > 1:
        img = tf.io.decode_jpeg(img_raw, channels=channels, ratio=ratio)
    else:
        img = tf.io.decode_jpeg(img_raw, channels=channels)
    img = tf.image.resize(img, target_size, method="bilinear")
    return tf.cast(img, tf.float32) / 255.0

def random_horizontal_flip(tensor_multi_ch, bbox, exist):
    """
    Performs horizontal flipping with 50% probability on multi-channel temporal tensor and bounding box.
    """
    do_flip = tf.random.uniform([]) > 0.5
    if do_flip:
        tensor_multi_ch = tf.image.flip_left_right(tensor_multi_ch)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        new_xmin = 1.0 - xmax
        new_xmax = 1.0 - xmin
        bbox = tf.stack([new_xmin, ymin, new_xmax, ymax])
    return tensor_multi_ch, bbox, exist

def load_norm_frame(path, target_size=(640, 640)):
    """
    Numpy-based frame loader for inference.
    """
    raw = cv2.imread(path)
    if raw is None:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    im = cv2.resize(raw, target_size)
    return im.astype(np.float32) / 255.0
