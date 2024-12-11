import logging
import os
from collections import defaultdict

import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 允许多版本 OpenMP 共存
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ObbTracker:
    """
    封装了yolo对象识别和deepsort追踪过程
    """
    def __init__(self, yolo_weight_path):
        self.yolo = initialize_yolo(yolo_weight_path)
        self.class_names = self.yolo.names
        self.tracker = DeepSort(max_age=20, n_init=3)
        # 类别已近出现对象的计数
        self.class_counters = defaultdict(int)
        # rack_id到obj_id的映射
        self.track2obj_map = {}

    def predict_and_track(self, frame, conf_threshold, specified_class_id=None):
        """
        返回[[检测框、obj_id、对象分类id], ...]，也就是说对象由 类别-obj_id 唯一标识
        检测框为xyxy或xyxyxyxyx的obb格式

        由于输入deepsort的检测框在obb的情况下不是原检测框，因此会为每个输入deepsort的检测框绑定一个哈希码（通过others字段）
        并将该哈希码映射到原检测框

        conf_threshold: 置信度阈值
        specified_class_id: 类别id，如果为None则不限制类别, 否则只追踪和展示指定类别
        """
        ## 1. 预测
        results = self.yolo(frame, verbose=False)[0]
        # 解析yolo预测结果: yolo预测的检测框，检测框的最小外接矩形框——用于追踪、类别id、置信度4个list
        predict_boxes, bboxes, cls_ls, conf_ls = parse_yolo_result(results)
        ## 2. 由检测输出构建追踪输入
        # 由于输入deepsort的检测框在obb的情况下不是原检测框，因此建立输入deepsort的检测框和原检测框映射
        # key为输入deepsort的检测框序号，value是原检测框
        xyxy2obb_map = {}
        detections = []
        # deepsort追踪接口的others入参可以为每个检测对象绑定信息，这里绑定其序号作为哈希码
        others = []
        for i in range(len(bboxes)):
            cls_id, conf, bbox = int(cls_ls[i]), conf_ls[i], bboxes[i]
            x1, y1, x2, y2 = map(int, bbox)

            # 当指定类别id时，只追踪指定类别
            if specified_class_id is not None and cls_id != specified_class_id:
                continue
            if conf < conf_threshold:
                continue

            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])
            others.append(i)
            xyxy2obb_map[i] = predict_boxes[i]
        ## 3. 追踪
        tracks = self.tracker.update_tracks(detections, frame=frame, others=others)
        ## 4. 由追踪框获取原检测框，并和obj_id、类别id一起封装返回
        res = []
        for i, track in enumerate(tracks):
            # 过滤追踪失败的对象
            if track.others is None:
                continue
            # 由追踪框获取原检测框
            box = xyxy2obb_map[track.others]

            class_id = track.get_det_class()
            obj_id = self.__track_id_to_obj_id(track.track_id, class_id)

            res.append([box, obj_id, class_id])
        return res

    def __track_id_to_obj_id(self, track_id, class_id):
        """
        obj_id是表示当前对象是所属类的第几个对象，，也就是说对象由 类别-obj_id 唯一标识，track_id则是追踪id，也可以唯一标识对象，但不区分类
        class_id: 当前对象的类别id
        """
        # 如果track_id不在映射中，则说明是新对象
        if track_id not in self.track2obj_map:
            self.class_counters[class_id] += 1
            self.track2obj_map[track_id] = self.class_counters[class_id]

        return self.track2obj_map[track_id]

def initialize_yolo(model_path):
    if not os.path.exists(model_path):
        logger.error(f"权重文件不存在：{model_path}")
        raise FileNotFoundError("Model weights file not found")

    yolo = YOLO(model_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    yolo.to(device)
    logger.info(f"使用 {device} 运行YOLO模型")
    return yolo

def parse_yolo_result(results):
    """
    解析yolo预测结果，返回yolo预测的检测框，检测框的最小外接矩形框、类别id、置信度4个list
    """
    if results.obb is not None:
        # deepsort算法的入参中检测框格式只能是xywh，将obb转换为其最小外接矩形的xyxy坐标
        bboxes = results.obb.xyxy.tolist()
        predict_boxes = results.obb.xyxyxyxy.tolist()
        cls_ls = results.obb.cls.tolist()
        conf_ls = results.obb.conf.tolist()
    else:
        bboxes = results.boxes.xyxy.tolist()
        predict_boxes = results.boxes.xyxy.tolist()
        cls_ls = results.boxes.cls.tolist()
        conf_ls = results.boxes.conf.tolist()
    return predict_boxes, bboxes, cls_ls, conf_ls