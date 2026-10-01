#trajectory exponential moving averafe (EMA) 

import numpy as np 

class TrajectoryEMAFilter:
    """
    giam thieu rung lac(filter) cua bounding box giua cac frame
    du dao quy dao quan tinh khi ma drone bi che khuat tam thoi 
    """
    def __init__(self, alpha=0.6,max_missing=15):
        """
        alpha(float): he so lam muot trong so frame hien tai
        max_missing(int): so frame toi da duy tri de bam duoi theo quan tinh truoc khi chuyeng sang trang thai Lost
        """
        self.alpha = alpha
        self.max_missing = max_missing
        self.smooth_box = None
        self.velocity = np.array([0.0, 0.0])
        self.missing_count = 0
        self.is_active = False

    def update(self, detected_box, is_detected):
        """
        cap nhat trang thai bo loc voi bounding box phat dien duoc [xmin, ymin, xmax, ymax]
        smooth_box: np.ndarray[4] or None
        state: "DETECTED | LOST"
        speed(float): do lon van toc tuc thi (pixel/frame)
        """
        if is_detected and detected_box is not None:
            cur_box = np.array(detected_box, dtype=np.float32)
            cur_cx = (cur_box[0] + cur_box[2]) / 2.0
            cur_cy = (cur_box[1] + cur_box[3]) / 2.0

            if self.smooth_box is None:
                self.smooth_box = cur_box
                self.velocity = np.array([0.0, 0.0])
            else:
                prev_cx = (self.smooth_box[0] + self.smooth_box[2]) / 2.0
                prev_cy = (self.smooth_box[1] + self.smooth_box[3]) / 2.0
                #cap nhat van toc di chuyen tuc thoi qua bo loc trung binh dong
                self.velocity = self.alpha * self.velocity + 0.4 * np.array([cur_cx - prev_cx, cur_cy - prev_cy])
                #lam muot bounding box
                self.smooth_box = self.alpha * cur_box + (1 - self.alpha) * self.smooth_box

            self.missing_count = 0
            self.is_active = True
            return self.smooth_box.copy(), "DETECTED", float(np.linalg.norm(self.velocity))
        else:
            #du bao quan tinh vi tri tiep theo khi mat dau tam thoi
            if self.is_active and self.missing_count < self.max_missing:
                self.missing_count += 1
                self.smooth_box[0] += self.velocity[0]
                self.smooth_box[1] += self.velocity[1]
                self.smooth_box[2] += self.velocity[0]
                self.smooth_box[3] += self.velocity[1]
                return self.smooth_box.copy(), "OCCLUDED", float(np.linalg.norm(self.velocity))
            else:
                self.is_active = False
                self.smooth_box = None
                return None, "LOST", 0.0