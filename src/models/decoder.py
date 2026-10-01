#bo giai ma maxpool 3x3 khong can NWS
import tensorflow as tf

def decode_detection(heatmap, offset, size, default_w=0.065, default_h=0.070, min_size=0.015):
    """
    decode bounding box tu heatmap bang maxpool3x3
    trich xuat diem cuc dai Top-K de phuc vu bai toan bam bat don muc tieu
    neu size du doan nho hon nguong min_size tu dong gan size prior trung binh cua drone trong dataset (default_w = 0.065, default_h = 0.07)
    dau ra:
    score: tensor size (B,) gia tri tin cay trong khoang [0, 1]
    bboxes: tensor size (B, 4) toa do bbox chuan hoa [xmin, ymin, xmax, ymax]
    """

    #loc local maximum 3x3
    hmax = tf.nn.max_pool2d(heatmap, ksize=3, strides=1, padding="SAME")
    keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
    peak_heatmap = heatmap * keep

    B = tf.shape(heatmap)[0]
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2]

    #trich xuat local maximum
    flat_peaks = tf.reshape(peak_heatmap, [B, H * W])  #tensor [B, 6400]
    top_scores, top_indices = tf.math.top_k(flat_peaks, k=1) #top_scores: [B, 1], top_indices: [B, 1]
    score = top_scores[:, 0] #tensor size [B] chua cac conf tung anh trong batch
    idx = top_indices[:, 0] #tensor size [B] chua cac chi so tam cua tung anh trong batch

    #vi du idx = [1640, 3220], W = 80
    grid_x = tf.cast(idx % W, tf.float32) #1640 % 80 = 40, 3220 % 80 = 20
    grid_y = tf.cast(idx // W, tf.float32) #1640 // 80 = 20, 3220 // 80 = 40
    #vi du 1640 -> [20, 0]

    #lay offset va size
    flat_offset = tf.reshape(offset, [B, H * W, 2])
    flat_size = tf.reshape(size, [B, H * W, 2])

    batch_indices = tf.range(B, dtype=tf.int32)
    gather_idx = tf.stack([batch_indices, idx], axis=-1)

    peak_offset = tf.gather_nd(flat_offset, gather_idx)
    peak_size = tf.gather_nd(flat_size, gather_idx)

    #tai tao toa do tam lien tuc tren grid chuan hoa [0, 1]
    cx = (grid_x + peak_offset[:, 0]) / tf.cast(W, tf.float32)
    cy = (grid_y + peak_offset[:, 1]) / tf.cast(H, tf.float32)

    #bo cuu kich thuoc an toan
    bw = peak_size[:, 0]
    bh = peak_size[:, 1]
    bw = tf.where(bw < min_size, tf.constant(default_w, dtype=bw.dtype), bw)
    bh = tf.where(bh < min_size, tf.constant(default_h, dtype=bh.dtype), bh)

    #chuyen doi sang toa do goc [xmin, ymin, xmax, ymax]
    xmin = tf.clip_by_value(cx - bw / 2.0, 0.0, 1.0)
    ymin = tf.clip_by_value(cy - bh / 2.0, 0.0, 1.0)
    xmax = tf.clip_by_value(cx + bw / 2.0, 0.0, 1.0)
    ymax = tf.clip_by_value(cy + bh / 2.0, 0.0, 1.0)

    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    return score, bboxes
# Alias de tuong thich ca 2 cach goi
decode_detections = decode_detection
