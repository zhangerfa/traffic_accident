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
        # 周围车辆信息：{frame_index: NearbyCar实例对象, ...}
        self.nearby_car_dict = {}

    def get_traffic_incidents(self, frame_index):
        """
        获取基于车辆和周围车辆关系可以判断的指定帧车辆匹配的交通事件列表
        """
        if self.__is_extract_frame_valid(frame_index):
            return None
        # 获取指定帧的周围车辆信息
        nearby_car = self.__get_nearby_car(frame_index)
        # todo:获取交通事件
        traffic_incidents = []
        return traffic_incidents

    def __get_nearby_car(self, frame_index):
        """
        获取指定帧的周围车辆信息
        """
        if frame_index in self.nearby_car_dict:
            return self.nearby_car_dict[frame_index]

        # 获取当前帧的车辆位置、速度
        box = self.get_pas_on_frame(frame_index)
        speed = self.get_speed_on_frame(frame_index)

        # todo:获取周围车辆信息
        nearby_car = NearbyCar(box, speed)
        self.nearby_car_dict[frame_index] = nearby_car


        return nearby_car

    def get_pas_on_frame(self, extract_frame):
        """
        获取指定帧的车辆位置
        """
        trace_index = self.__get_trace_index_by_frame(extract_frame)
        if trace_index is None:
            return None

        return self.trace_ls[trace_index][0]

    def get_speed_on_frame(self, extract_frame, time_span=10):
        """
        获取指定帧的车辆速度矢量
        """
        start_index = self.__get_trace_index_by_frame(extract_frame)
        if start_index is None:
            return None

        if len(self.trace_ls[start_index:]) < time_span:
            logger.error(f"Error: 当前车辆轨迹数据为空或者不包含第{extract_frame}帧的轨迹数据")
            return None

        # 由最后一帧和倒数第time_span帧的位置计算速度矢量
        start_box = self.trace_ls[start_index - time_span][0]
        end_box = self.trace_ls[start_index][0]
        start_x, start_y, _, _, _ = start_box
        end_x, end_y, _, _, _ = end_box
        # 速度矢量
        speed_vector = [(end_x - start_x) / time_span, (end_y - start_y) / time_span]

        return speed_vector

    def __get_trace_index_by_frame(self, frame_index):
        """
        通过帧数获取轨迹数据的索引，因为轨迹数据是按照帧数递增的，所以可以直接通过帧数获取索引
        """
        if self.__is_extract_frame_valid(frame_index):
            return None
        max_frame = self.__get_max_frame()
        start_index = max_frame - frame_index
        return start_index

    def __get_max_frame(self):
        """
        获取轨迹中最大的帧数
        """
        if len(self.trace_ls) == 0:
            return -1
        return self.trace_ls[-1][1]

    def __is_extract_frame_valid(self, extract_frame):
        """
        判断提取帧是否有效
        """
        if extract_frame < 0 or self.__get_max_frame() < extract_frame:
            logger.error(f"Error: 当前车辆轨迹数据为空或者不包含第{extract_frame}帧的轨迹数据")
            return False
        return True

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

class NearbyCar:
    """
    存储一辆车周围车辆的信息
    """
    def __init__(self, box, speed):
        # 车辆位置
        self.pos = box
        # 车辆速度
        self.speed = speed
        # 前车，类型为Car实例对象，下同
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