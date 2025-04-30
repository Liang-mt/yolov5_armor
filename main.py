import onnxruntime
import numpy as np
import cv2
import torch
from collections import deque


class ArmorZoomTracker:
    def __init__(self, frame_shape):
        # 初始化跟踪器参数
        self.frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)  # 图像中心坐标(x, y)
        self.frame_size = frame_shape[:2]  # 原始图像尺寸(height, width)

        # 可调节的缩放参数
        self.max_zoom = 2.0          # 最大缩放倍数
        self.scale_factor = 0.45     # 面积比例因子
        self.ratio_exponent = 0.3    # 面积指数调整
        self.zoom_speed = 0.1        # 缩放平滑速度
        self.history_length = 5      # 位置历史长度

        # 跟踪状态变量
        self.current_zoom = 1.0       # 当前缩放倍数
        self.target_zoom = 1.0        # 目标缩放倍数
        self.position_history = deque(maxlen=self.history_length)  # 位置历史队列
        self.lost_counter = 0         # 目标丢失计数器
        self.max_lost_frames = 15     # 最大允许丢失帧数

    def update_armor(self, armor_rect):
        """更新装甲板位置并计算目标缩放倍数"""
        x, y, w, h = armor_rect
        # 将装甲板中心加入历史记录
        self.position_history.append((x + w // 2, y + h // 2))

        # 根据装甲板面积计算缩放倍数
        armor_area = w * h
        frame_area = self.frame_size[0] * self.frame_size[1]
        area_ratio = armor_area / frame_area

        # 调整后的面积比例和缩放倍数计算
        adjusted_ratio = area_ratio ** self.ratio_exponent
        self.target_zoom = np.clip(1.0 / (adjusted_ratio * self.scale_factor),
                                   1.0, self.max_zoom)
        self.lost_counter = 0  # 重置丢失计数器

    def smooth_zoom(self):
        """平滑过渡当前缩放倍数"""
        self.current_zoom += (self.target_zoom - self.current_zoom) * self.zoom_speed
        return np.clip(self.current_zoom, 1.0, self.max_zoom)

    def get_zoomed_frame(self, frame):
        """获取缩放后的画面及裁剪信息"""
        if self.current_zoom == 1.0:
            return frame, (0, 0, *self.frame_size[::-1]), 1.0

        # 计算裁剪区域尺寸
        zoom = self.current_zoom
        new_w = int(self.frame_size[1] / zoom)  # 裁剪区域宽度
        new_h = int(self.frame_size[0] / zoom)  # 裁剪区域高度

        # 计算平均跟踪位置（使用历史位置平滑）
        avg_center = np.mean(self.position_history, axis=0) if self.position_history else self.frame_center

        # 计算裁剪区域坐标（确保不越界）
        x = int(np.clip(avg_center[0] - new_w // 2, 0, self.frame_size[1] - new_w))
        y = int(np.clip(avg_center[1] - new_h // 2, 0, self.frame_size[0] - new_h))

        # 执行裁剪和缩放
        cropped = frame[y:y + new_h, x:x + new_w]
        zoomed_frame = cv2.resize(cropped, (self.frame_size[1], self.frame_size[0]))  # 保持原始分辨率
        return zoomed_frame, (x, y, new_w, new_h), zoom


class yolov5_armor:
    def __init__(self):
        # 模型参数初始化
        self.conf_thres = 0.3  # 置信度阈值
        self.iou_thres = 0.45  # IOU阈值
        self.imgsz = (384, 640)  # 模型输入尺寸(height, width)
        self.names = ['BG', 'B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BBs', 'BBb',
                      'RG', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RBs', 'RBb',
                      'NG', 'N1', 'N2', 'N3', 'N4', 'N5', 'NO', 'NBs', 'NBb',
                      'PG', 'P1', 'P2', 'P3', 'P4', 'P5', 'PO', 'PBs', 'PBb']

        # ONNX模型初始化
        self.weight = "./weights/model-opt-4.onnx"
        self.session = onnxruntime.InferenceSession(self.weight, None)
        self.tracker = None  # 缩放跟踪器

    def scale_coords(self, img1_shape, coords, img0_shape):
        """将检测坐标从模型输入尺寸缩放回原始图像尺寸"""
        # 计算缩放比例和填充
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

        # 调整坐标
        coords[:, [0, 2, 4, 6]] -= pad[0]  # x坐标调整
        coords[:, [1, 3, 5, 7]] -= pad[1]  # y坐标调整
        coords[:, :8] /= gain  # 缩放坐标
        return coords

    def preprocess(self, im):
        """图像预处理"""
        im0 = im.copy()  # 保留原始图像
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 转RGB
        if im.shape[:2] != self.imgsz:
            im = cv2.resize(im, (self.imgsz[1], self.imgsz[0]))  # 缩放到模型输入尺寸
        return im0, np.expand_dims(im, axis=0).astype('float32')

    def postprocess(self, pred_raw, im0):

        pred = []
        for p in pred_raw:
            mask = p[..., 8] > self.conf_thres
            p = p[mask]
            if p.shape[0] > 0:
                xmin = torch.min(p[..., [0, 2, 4, 6]], dim=1).values.int().numpy()
                xmax = torch.max(p[..., [0, 2, 4, 6]], dim=1).values.int().numpy()
                ymin = torch.min(p[..., [1, 3, 5, 7]], dim=1).values.int().numpy()
                ymax = torch.max(p[..., [1, 3, 5, 7]], dim=1).values.int().numpy()
                bbox = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, x2, y1, y2 in
                        zip(xmin, xmax, ymin, ymax)]
                conf = [float(c) for c in p[..., 8].numpy()]
                cls_color = torch.argmax(p[..., 9:13], dim=-1).numpy()
                cls_number = torch.argmax(p[..., 13:22], dim=-1).numpy()
                cls = cls_color * 9 + cls_number
                ids = cv2.dnn.NMSBoxes(bbox, conf, self.conf_thres, self.iou_thres)
                p = torch.stack([
                    torch.cat([
                        torch.tensor(p[i, :8]).float(),
                        torch.tensor([conf[i]]).float(),
                        torch.tensor([cls[i]]).float()
                    ], dim=0)
                    for i in ids.reshape(ids.shape[0])
                ], dim=0)
            pred.append(p)
            # 用于存放结果
        detections = []

        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :8] = self.scale_coords(self.imgsz, det[:, :8], im0.shape).round()

                for d in det:
                    pt0 = (int(d[0]), int(d[1]))
                    pt1 = (int(d[2]), int(d[3]))
                    pt2 = (int(d[4]), int(d[5]))
                    pt3 = (int(d[6]), int(d[7]))
                    position = (pt0, pt1, pt2, pt3)
                    cls = self.names[int(d[-1])]
                    conf = round(torch.sigmoid(d[8]).item(), 2)

                    # 按字典形式存入列表，方便调用
                    detections.append({'cls': cls, 'conf': conf, 'position': position})

        return detections

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        """在图像上绘制检测框"""
        for d in detections:
            pts = np.array(d['position'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)  # 绘制四边形
            # 显示类别和置信度
            cv2.putText(frame, f"{d['cls']} {d['conf']}",
                        pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def detect(self, img):
        # 预处理和推理
        im0, img = self.preprocess(img)
        input_name = self.session.get_inputs()[0].name
        pred = torch.tensor(self.session.run(None, {input_name: img})[0])

        # 后处理得到检测结果
        detections = self.postprocess(pred, im0)
        # 初始化跟踪器
        if self.tracker is None:
            self.tracker = ArmorZoomTracker(im0.shape)

        # 更新跟踪器状态
        main_rect = None
        if detections:
            # 选择面积最大的检测作为跟踪目标
            main_det = max(detections, key=lambda d:
            (d['position'][2][0] - d['position'][0][0]) *
            (d['position'][2][1] - d['position'][0][1]))
            pts = np.array(main_det['position'])
            x, y, w, h = cv2.boundingRect(pts)
            main_rect = (x, y, w, h)
            self.tracker.update_armor(main_rect)
        else:
            self.tracker.lost_counter += 1
            # 超过最大丢失帧数时重置缩放
            if self.tracker.lost_counter > self.tracker.max_lost_frames:
                self.tracker.target_zoom = 1.0

        # 应用平滑缩放
        self.tracker.smooth_zoom()
        # 获取缩放后的画面和裁剪信息
        zoomed_frame, crop_info, _ = self.tracker.get_zoomed_frame(im0)

        # 在原始图像上绘制检测框
        original_frame = self.draw_detections(im0.copy(), detections, (0, 255, 0))

        # 转换检测框坐标到缩放后的坐标系
        zoomed_detections = []
        if crop_info[2] > 0 and crop_info[3] > 0:  # 确保有效裁剪区域
            frame_height, frame_width = im0.shape[:2]
            crop_x, crop_y, new_w, new_h = crop_info
            # 计算缩放比例
            scale_x = frame_width / new_w
            scale_y = frame_height / new_h

            for d in detections:
                new_pts = []
                for (x, y) in d['position']:
                    # 坐标转换公式：
                    # 缩放后坐标 = (原始坐标 - 裁剪起点) * 缩放比例
                    dx = (x - crop_x) * scale_x
                    dy = (y - crop_y) * scale_y
                    new_pts.append((int(dx), int(dy)))
                zoomed_detections.append({
                    'cls': d['cls'],
                    'conf': d['conf'],
                    'position': new_pts
                })

        # 在缩放图像上绘制转换后的检测框
        zoomed_frame = self.draw_detections(zoomed_frame, zoomed_detections, (0, 0, 255))

        # 添加调试信息
        cv2.putText(original_frame, f"Detections: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(zoomed_frame, f"Zoom: {self.tracker.current_zoom:.1f}x", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return original_frame, zoomed_frame


if __name__ == "__main__":
    video_path = "./video/3.mp4"
    armor = yolov5_armor()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        original, zoomed = armor.detect(frame)

        # 显示结果
        cv2.imshow("Original Detection", original)
        cv2.imshow("Zoomed Detection", zoomed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()