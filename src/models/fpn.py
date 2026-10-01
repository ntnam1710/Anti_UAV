#feature pyramid network
import tensorflow as tf
from tensorflow.keras import layers
from .attention import CoordinateAttention

def fpn(c2, c3, c4, c5, fpn_dim=128):
    """
    c2 160x150x256
    c3 80x80x512
    c4 40x40x1024
    c5 20x20x2048

    p2 160x150x128
    p3 80x80x128
    p4 40x40x128
    p5 20x20x128
    """
    #cac tang tich chap 1x1 giam so channel xuong fpn_dim
    lat_c5 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c5")(c5)
    lat_c4 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c4")(c4)
    lat_c3 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c3")(c3)
    lat_c2 = layers.Conv2D(fpn_dim, kernel_size=1, name="fpn_lat_c2")(c2)

    #lan truyen tu tren xuong (top down pathway)
    p5 = lat_c5
    p4 = layers.Add(name="fpn_add_p4")([lat_c4, layers.UpSampling2D(size=2, name="fpn_upsample_p5")(p5)])
    p3 = layers.Add(name="fpn_add_p3")([lat_c3, layers.UpSampling2D(size=2, name="fpn_upsample_p4")(p4)])
    p2 = layers.Add(name="fpn_add_p2")([lat_c2, layers.UpSampling2D(size=2, name="fpn_upsamling_p3")(p3)])

    #tang conv 3x3 de lam muot feature
    p5 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p5")(p5)
    p4 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p4")(p4)
    p3 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p3")(p3)
    p2 = layers.Conv2D(fpn_dim, kernel_size=3, padding="same", name="fpn_smooth_p2")(p2)

    #nhung Coordinate Attention tai 2 tang co do phan giai chi tiet cao la p2 va p3
    p2 = CoordinateAttention(name="ca_p2")(p2)
    p3 = CoordinateAttention(name="ca_p3")(p3)

    #p4 va p5 duoc giu nguyen khong su dung attention vi chung chua cac thanh phan tinh toan vi tri cua object

    return p2,p3,p4,p5