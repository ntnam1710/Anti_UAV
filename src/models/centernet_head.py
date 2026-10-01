#anchor free 
import tensorflow as tf
from tensorflow.keras import layers, Model
from .backbone import resnet50v2_backbone
from .fpn import fpn

@tf.keras.utils.register_keras_serializable(package="model_tf", name="AnchorFreeCenterHead")
class AnchorFreeCenterHead(layers.Layer):
    """
    head layer Centerhead anchor-free tach biet thanh 3 nhanh chuyen biet:
    -nhanh 1: heatmap(80x80x1) de du doan xac suat tam cua drone tren tung o luoi
    -nhanh 2: offset(80x80x2) de du doan sai lech offset cua tam so voi diem goc
    -nhanh 3: size(80x80x2) de du doan size rong-cao 
    """

    def __init__(self, fpn_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.fpn_dim = fpn_dim

    def build(self, input_shape):
        #1 nhanh heatmap
        self.hm_conv1 = layers.Conv2D(128, 3, padding="same", activation="relu", name="hm_conv1")
        self.hm_bn1 = layers.BatchNormalization(name="hm_bn1")
        self.hm_out = layers.Conv2D(
            1, 1, padding="same", activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-4.595),
            dtype="float32", name="heatmap"
        )

        #2 nhanh offset
        self.off_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu", name="off_conv1")
        self.off_bn1 = layers.BatchNormalization(name="off_bn1")
        self.off_out = layers.Conv2D(2, 1, padding="same", dtype="float32", name="offset")

        #3 nhanh size(w, h)
        self.size_conv1 = layers.Conv2D(64, 3, padding="same", activation="relu", name="size_conv1")
        self.size_bn1 = layers.BatchNormalization(name="size_bn1")
        self.size_out = layers.Conv2D(
            2, 1, padding="same", activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.66),
            dtype="float32", name="size"
        )

        super().build(input_shape)

    def call(self, x):
        #pred heatmap
        h_hm = self.hm_bn1(self.hm_conv1(x))
        pred_hm = self.hm_out(h_hm)

        #pred offset
        h_off = self.off_bn1(self.off_conv1(x))
        pred_off = self.off_out(h_off)

        #pred size
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

def build_detection_model(input_shape=(640, 640, 9), fpn_dim=128, name="ResNet50v2_FPN"):
    """
    dau vao: tensor chuoi temporal 9 channels (t-1, t, t+1) kich thuoc (B, 640, 640, 9)
    dau ra: dict chua 'heatmap', 'offsize', 'size' o do phan giai stride 8 (80x80)
    """
    inputs = layers.Input(shape=input_shape, name="temporal_input")
    c2, c3, c4, c5 = resnet50v2_backbone(inputs) #lay backbone
    p2, p3, p4, p5 = fpn(c2, c3, c4, c5, fpn_dim=fpn_dim) #lay fpn

    #hop nhat cac tang p2, p3, p4, p5 ve luoi stride 8
    p2_down = layers.MaxPool2D(pool_size=2, name="fuse_p2_down")(p2)
    p3_same = p3
    p4_up = layers.UpSampling2D(size=2, name="fuse_p4_up")(p4)
    p5_up = layers.UpSampling2D(size=4, name="fuse_p5_up")(p5)

    fused = layers.Concatenate(axis=-1, name="fuse_concat")([p2_down, p3_same, p4_up, p5_up])
    fused = layers.BatchNormalization(name="fuse_bn")(
        layers.Conv2D(fpn_dim, kernel_size=3, padding="same", activation="relu", name="fuse_conv")(fused)
    )

    outputs = AnchorFreeCenterHead(fpn_dim=fpn_dim, name="anchor_free_head")(fused)
    
    return Model(inputs=inputs, outputs=outputs, name=name)

def load_trained_model(checkpoint_path, input_shape=(640, 640, 9), fpn_dim=128):
    """
    Nap mo hinh da huan luyen tu tep .keras.
    Uu tien su dung tf.keras.models.load_model voi day du custom_objects.
    Neu gap loi cau truc se tu dong chuyen sang khoi tao kien truc moi roi nap trong so.
    """
    try:
        from .attention import CoordinateAttention
        return tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                "CoordinateAttention": CoordinateAttention,
                "AnchorFreeCenterHead": AnchorFreeCenterHead
            },
            compile=False
        )
    except Exception:
        model = build_detection_model(input_shape=input_shape, fpn_dim=fpn_dim)
        model.load_weights(checkpoint_path)
        return model