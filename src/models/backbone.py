#backbone resnet50v2 

import tensorflow as tf
from tensorflow.keras import layers

def bottleneck_v2(x, filters, stride=1, projection=False, name="block"):
    x_pre = layers.Activation("relu", name=f"{name}_pre_relu")(
        layers.BatchNormalization(name=f"{name}_pre_bn")(x)
    )

    if projection:
        shortcut = layers.Conv2D(filters * 4, kernel_size=1, strides=stride, use_bias=False, name=f"{name}_proj")(x_pre)
    elif stride > 1:
        shortcut = layers.MaxPool2D(pool_size=1, strides=stride, name=f"{name}_pooling")(x_pre)
    else:
        shortcut = x

    c1 = layers.Activation("relu", name=f"{name}_relu")(
        layers.BatchNormalization(name=f"{name}_bn1")(
            layers.Conv2D(filters, kernel_size=1, strides=1, use_bias=False, name=f"{name}_conv1")(x_pre)
        )
    )

    c2 = layers.Activation("relu", name=f"{name}_relu2")(
        layers.BatchNormalization(name=f"{name}_bn2")(
            layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv2")(c1)
        )
    )

    c3 = layers.Conv2D(filters * 4, kernel_size=1, strides=1, use_bias=False, name=f"{name}_conv3")(c2)
    return layers.Add(name=f"{name}_add")([shortcut, c3])

def resnet50v2_backbone(input_tensor):
    """
    c2: stride 4 160x160x256 - chi tiet duong net sac net
    c3: stride 8 80x80x512 - luoi trung tam de du doan vat the
    c4: stride 16 40x40x1024 - thong tin vi tri tam trung
    c5: stride 32 20x20x2048 - ngu canh toan cuc de nhan biet dau la nen, dau la object
    """

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False, name="stem_conv")(input_tensor)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="stem_pool")(
        layers.Activation("relu", name="stem_relu")(
            layers.BatchNormalization(name="stem_bn")(x)
        )
    )

    #block 1 (stage 1) -> xuat c2
    x = bottleneck_v2(x, 64, stride=1, projection=True, name="stage1_b1")
    x = bottleneck_v2(x, 64, stride=1, projection=False, name="stage1_b2")
    c2 = bottleneck_v2(x, 64, stride=1, projection=False, name="stage1_b3")

    #block 2 (stage 2) -> xuat c3
    x = bottleneck_v2(x, 128, stride=2, projection=True, name="stage2_b1")
    x = bottleneck_v2(x, 128, stride=1, projection=False, name="stage2_b2")
    x = bottleneck_v2(x, 128, stride=1, projection=False, name="stage2_b3")
    c3 = bottleneck_v2(x, 128, stride=1, projection=False, name="stage2_b4")

    #block 3 (stage 3) -> xuat c4
    x = bottleneck_v2(c3, 256, stride=2, projection=True, name="stageb3_b1")
    for i in range(2, 7):
        x = bottleneck_v2(x, 256, stride=1,  projection=False, name=f"stage3_b{i}")
    c4 = x

    #block 4 (stage 4) -> xuat c5
    x = bottleneck_v2(c4, 512, stride=2, projection=True, name="stage4_b1")
    x = bottleneck_v2(x, 512, stride=1, projection=False, name="stage4_b2")
    c5 = layers.Activation("relu", name="post_relu")(
        layers.BatchNormalization(name="post_bn")(
            bottleneck_v2(x, 512, stride=1, projection=False, name="stage4_b3")
        )
    )
    return c2, c3, c4, c5
