import onnxruntime
import numpy as np
import cv2
import torch


class yolov5_armor():
    def __init__(self):
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.imgsz = (384,640)

        self.names = ['BG', 'B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BBs', 'BBb',
                      'RG', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RBs', 'RBb',
                      'NG', 'N1', 'N2', 'N3', 'N4', 'N5', 'NO', 'NBs', 'NBb',
                      'PG', 'P1', 'P2', 'P3', 'P4', 'P5', 'PO', 'PBs', 'PBb']

        self.weight = "./weights/model-opt-4.onnx"
        self.session = onnxruntime.InferenceSession(self.weight, None)

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
        coords[:, :8] /= gain
        # clip_coords(coords, img0_shape)
        return coords

    def preprocess(self,im):

        im0 = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB

        if im.shape[1] != 640 or im.shape[0] != 384:
            im = cv2.resize(im, (640, 384))  # 调整图像尺寸为 640x384
        im = np.expand_dims(im, axis=0)
        img = im.astype('float32')

        return im0, img

    def postprocess(self,pred_raw,im0):

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
                    position = (pt0,pt1,pt2,pt3)
                    cls = self.names[int(d[-1])]
                    conf = round(torch.sigmoid(d[8]).item(), 2)

                    # 按字典形式存入列表，方便调用
                    detections.append({'cls': cls, 'conf': conf, 'position': position})

        return detections

    def detectdraw(self,frame, dict_list):

        if dict_list is None:
            return frame
        for detection in dict_list:
            cls = detection['cls']
            conf = detection['conf']
            pt0, pt1, pt2, pt3 = detection['position']

            cv2.line(frame, pt0, pt1, (0, 255, 0), 2)
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.line(frame, pt2, pt3, (0, 255, 0), 2)
            cv2.line(frame, pt3, pt0, (0, 255, 0), 2)

            cv2.putText(frame, cls, pt0, 1, 1, (0, 255, 0))
            cv2.putText(frame, str(conf), pt3, 1, 1, (0, 255, 0))


        return frame

    def detect(self,img):
        im0, img = self.preprocess(img)
        input_name = self.session.get_inputs()[0].name
        pred_raw = torch.Tensor(self.session.run(None, {input_name: img})[0])  # 执行推理
        detection = self.postprocess(pred_raw,im0)

        img = self.detectdraw(im0,detection)

        return img


if __name__ == "__main__":
    video_path = "./video/1.mp4"  # 视频文件路径
    armor = yolov5_armor()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()

        if success:
            im0 = armor.detect(frame)

            cv2.imshow("Video", im0)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频流和窗口
    cap.release()
    cv2.destroyAllWindows()
