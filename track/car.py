"""
存储车辆数据及计算逻辑
"""
import logging

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Car:
    def __init__(self, id, class_id):
        # 车辆id
        self.id = id
        # 车辆类别id
        self.class_id = class_id
        # 车辆轨迹：[[xywhθ格式的坐标, frame_index], ...]，轨迹中数据的顺序随着frame_index的增加而增加
        self.trace_ls = []
        # 前车
        self.pre_car = None
        # 后车
        self.behind_car = None
        # 左前车
        self.left_pre_car = None
        # 右前车
        self.right_pre_car = None
        # 左后车
        self.left_behind_car = None
        # 右后车
        self.right_behind_car = None

    def get_speed_on_frame(self, extract_frame, time_span=10):
        """
        获取指定帧的车辆速度矢量
        """
        max_frame = self.trace_ls[-1][1]
        if len(self.trace_ls) == 0 or max_frame < extract_frame:
            return None
        start_index = max_frame - extract_frame
        if len(self.trace_ls[start_index:]) < time_span:
            return None

        # 由最后一帧和倒数第time_span帧的位置计算速度矢量
        start_box = self.trace_ls[start_index - time_span][0]
        end_box = self.trace_ls[start_index][0]
        start_x, start_y, _, _, _ = start_box
        end_x, end_y, _, _, _ = end_box
        # 速度矢量
        speed_vector = [(end_x - start_x) / time_span, (end_y - start_y) / time_span]

        return speed_vector

    @classmethod
    def is_obb(cls, box):
        """
        判断box是否为obb格式
        """
        if len(box) != 4:
            return False
        for point in box:
            if len(point) != 2:
                return False
        return True

    @classmethod
    def is_xywhangle(cls, box):
        """
        判断box是否为xywhθ格式
        """
        if len(box) != 5:
            return False
        return True

    @classmethod
    def xywhangle_to_obb(cls, xywhangle):
        """
        将xywhθ转换为obb（xyxyxyxy）格式
        """
        if not cls.is_xywhangle(xywhangle):
            return None
        x, y, w, h, theta = xywhangle
        rotated_box = cv2.boxPoints(((x, y), (w, h), theta))  # 获取旋转矩形的四个顶点

        return rotated_box