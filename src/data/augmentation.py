"""
viet cac ham aug va ham preprocess
"""

import tensorflow as tf
import numpy as np 
import cv2


def decode_and_resize_img(file_path, channels=3, target_size=(640, 640), ratio=1):
    """
    decode anh tu file path va resize ve target_size
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
    ham lat ngang anh va bbox
    tensor_multi_ch: chuoi tensor thoi gian 
    bbox: la list toa do cua bbox goc
    exist: (bool) trang thai xem co drone trong anh hay khong
    """
    do_flip = tf.random.uniform([]) > 0.5
    if do_flip:
        tensor_multi_ch = tf.image.flip_left_right(tensor_multi_ch)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        new_xmin = 1.0 - xmax 
        new_xmax = 1.0 - xmin
        bbox = tf.stack([new_xmin, ymin, new_xmax, ymax])
    return tensor_multi_ch, bbox, exist
    