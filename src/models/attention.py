#tao lop Coordinate Attention 

import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="model_tf", name="CoordinateAttention")
class CoordinateAttention(layers.Layer):
    """
    phan da phep gom cum 2D thanh 2 phep tinh 1D theo chieu ngang-doc
    giup mang noron dinh vi chinh xac toa do cua muc tieu drone nho
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

        #gom cum trung binh toan cuc 1D theo chieu doc-ngang
        #z_c^h(h) = (1/W) * sum(x) x thuoc h
        x_h = tf.reduce_mean(x, axis=2, keepdims=True) #kich thuoc: (B, H, 1, C)
        #z_c^w(w) = (1/H) * sum(x) x thuoc w
        x_w = tf.reduce_mean(x, axis=1, keepdims=True) #kich thuoc: (B, 1, W, C)
        x_w_perm = tf.transpose(x_w, perm=[0, 2, 1, 3]) #chuen vi ma tran thanh (B, W, 1, C)

        #ghep noi theo truc khong gian va giam chieu channel
        concat = tf.concat([x_h, x_w_perm], axis=1) #kich thuoc: (B, H+W, 1, C)
        y = self.act(self.bn(self.conv_shared(concat))) #kich thuoc: (B, H+W, 1, C//r)

        #tach ve lai thanh 2 vector 1D doc-ngang
        y_h = y[:, :h, :, :] #kich thuoc (B, H, 1, C//r)
        y_w = tf.transpose(y[:, h:, :, :], perm=[0, 2, 1, 3]) #kich thuoc (B, 1, W, C//r)
        
        #tao weight chu y sigmoid cho tung toa do
        att_h = self.conv_h(y_h) #kich thuoc: (B, H, 1, C)
        att_w = self.conv_w(y_w) #kich thuoc: (B, 1, W, C)

        #tai can chinh features dau vao: y_c(i, j) = x_c(i, j) * g_c^h(i) * g_c^w(j)
        return x * att_h * att_w

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config