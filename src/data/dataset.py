"""
tao ham dataloader tao thanh chuoi triplets
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from .augmentation import decode_and_resize_img, random_horizontal_flip

def build_preprocess_fn(voc_root, target_size=(640, 640), is_training=True):
    """
    voc_root(str): duong dan folder goc chua dataset dang VOC
    target_size(tuple): kich thuoc cua anh dau vao
    is_training(bool): flag de xac dinh che do huan luyen(bat augmentation)
    return:
        ham anh xa tien xu ly nhan duong dan anh va nhan BB
    """

    orig_w, orig_h = 1920.0, 1080.0
    grid_h, grid_w = target_size[0] // 8, target_size[1] // 8 #luoi dac trung stride 8

    def preprocess_sample(vis_tm1_rel, vis_t_rel, vis_tp1_rel, xmin, ymin, xmax, ymax, exist):
        #noi path vao folder goc
        p_vis_tm1 = tf.strings.join([voc_root, "/", vis_tm1_rel])
        p_vis_t = tf.strings.join([voc_root, "/", vis_t_rel])
        p_vis_tp1 = tf.strings.join([voc_root, "/", vis_tp1_rel])

        #decode va chuan hoa ve target size
        img_vis_tm1 = decode_and_resize_img(p_vis_tm1, channels=3, target_size=target_size)
        img_vis_t = decode_and_resize_img(p_vis_t, channels=3, target_size=target_size)
        img_vis_tp1 = decode_and_resize_img(p_vis_tp1, channels=3, target_size=target_size)

        #gop 3 frames lien tiep thanh 1 tensor 9 kenh
        tensor_input = tf.concat([img_vis_tm1, img_vis_t, img_vis_tp1], axis=-1)

        #chuan hoa BB ve 0-1
        norm_xmin = tf.clip_by_value(tf.cast(xmin, tf.float32) / orig_w, 0.0, 1.0)
        norm_ymin = tf.clip_by_value(tf.cast(ymin, tf.float32) / orig_h, 0.0, 1.0)
        norm_xmax = tf.clip_by_value(tf.cast(xmax, tf.float32) / orig_w, 0.0, 1.0)
        norm_ymax = tf.clip_by_value(tf.cast(ymax, tf.float32) / orig_h, 0.0, 1.0)

        norm_box = tf.where(
            exist == 1,
            tf.stack([norm_xmin, norm_ymin, norm_xmax, norm_ymax]),
            tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        )

        #augmentation
        if is_training:
            tensor_input, norm_box, exist = random_horizontal_flip(tensor_input, norm_box, exist)

        #tao nhan gaussian heatmap tren grid
        xs = tf.cast(tf.range(grid_w), tf.float32)
        ys = tf.cast(tf.range(grid_h), tf.float32)
        grid_x, grid_y = tf.meshgrid(xs, ys)

        cur_xmin, cur_ymin = norm_box[0], norm_box[1]
        cur_xmax, cur_ymax = norm_box[2], norm_box[3]

        #tao toa do tam va size bbox
        cx = (cur_xmin + cur_xmax) / 2.0
        cy = (cur_ymin + cur_ymax) / 2.0
        bw = cur_xmax - cur_xmin
        bh = cur_ymax - cur_ymin

        #toa do tam tren grid 
        px = cx * float(grid_w)
        py = cy * float(grid_h)
        
        ix = tf.clip_by_value(tf.cast(tf.math.floor(px), tf.int32), 0, grid_w - 1)
        iy = tf.clip_by_value(tf.cast(tf.math.floor(py), tf.int32), 0, grid_h - 1)

        #do lech tam pixel
        dx = px - tf.cast(ix, tf.float32)
        dy = py - tf.cast(iy, tf.float32)

        #ban kinh gaussian vao kich thuoc doi tuong
        sigma = tf.maximum(1.0, tf.sqrt(tf.maximum(bw * float(grid_w) * bh * float(grid_h), 1e-4)) / 3.0)
        dist_sq = tf.square(grid_x - tf.cast(ix, tf.float32)) + tf.square(grid_y - tf.cast(iy, tf.float32))
        gaussian = tf.exp(-dist_sq / (2.0 * tf.square(sigma)))
        gaussian = tf.expand_dims(gaussian, axis=-1)

        #mask nhi pha xac dinh vi tri o luoi chua tam doi tuong
        center_point_mask = tf.cast(
            tf.logical_and(tf.equal(tf.cast(grid_x, tf.int32), ix), tf.equal(tf.cast(grid_y, tf.int32), iy)),
            tf.float32
        )

        center_point_mask = tf.expand_dims(center_point_mask, axis=-1)
        center_point_mask = tf.where(exist == 1, center_point_mask, tf.zeros_like(center_point_mask))
        
        #phan phoi xac xuat
        heatmap = tf.where(exist == 1, gaussian, tf.zeros_like(gaussian))
        offset_map = tf.concat([center_point_mask * dx, center_point_mask * dy], axis=-1)
        size_map = tf.concat([center_point_mask * bw, center_point_mask * bh], axis=-1)

        labels = {
            "heatmap": heatmap,
            "offset": offset_map,
            "size": size_map,
            "mask": center_point_mask,
            "raw_bbox": tf.cast(norm_box, tf.float32),
            "bbox_output": tf.cast(norm_box, tf.float32)
        }

        return tensor_input, labels

    return preprocess_sample

def create_temporal_dataset(csv_path, voc_root, batch_size=8, target_size=(640, 640), is_training=True, shuffle_buffer=1024):
    """
    Khởi tạo pipeline tf.data.Dataset cho chuỗi Temporal Triplet
    
    Tham số:
        csv_path (str): Đường dẫn tệp CSV chứa danh sách bộ ba khung hình.
        voc_root (str): Đường dẫn thư mục gốc dataset VOC.
        batch_size (int): Kích thước mini-batch.
        target_size (tuple): Kích thước ảnh đầu vào (H, W).
        is_training (bool): Chế độ huấn luyện (bật xáo trộn dữ liệu).
        shuffle_buffer (int): Kích thước bộ đệm xáo trộn.
        
    Trả về:
        tuple: (tf.data.Dataset, int tổng số mẫu dữ liệu).
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy têp CSV tại path: {csv_path}")
    df = pd.read_csv(csv_path)
    total_samples = len(df)

    ds = tf.data.Dataset.from_tensor_slices((
        df["vis_tm1"].values.astype(str),
        df["vis_t"].values.astype(str),
        df["vis_tp1"].values.astype(str),
        df["xmin"].values.astype(np.float32),
        df["ymin"].values.astype(np.float32),
        df["xmax"].values.astype(np.float32),
        df["ymax"].values.astype(np.float32),
        df["exist"].values.astype(np.int32)
    ))

    if is_training:
        ds = ds.shuffle(buffer_size=min(total_samples, shuffle_buffer), reshuffle_each_iteration=True)

    preprocess_fn = build_preprocess_fn(voc_root, target_size=target_size, is_training=is_training)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds, total_samples