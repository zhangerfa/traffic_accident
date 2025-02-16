"""
存储车辆的历史轨迹数据，以及特定帧的周围车辆信息
"""
import logging
import math

import cv2

from incident.Incident import Incident

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Car:
    def __init__(self, id, class_id):
        # 车辆id
        self.id = id
        # 车辆类别id
        self.class_id = class_id
        # 车辆长度
        self.length = None
        # 车辆宽度
        self.width = None
        # 车辆历史轨迹：[[xywhθ格式的坐标, frame_index], ...]，轨迹中数据的顺序随着frame_index的增加而增加
        self.trace_ls = []
        # 车辆在不同帧的沿着道路行驶方向的车辆信息缓存：{frame_index: 对应帧的CarAlongLane实例对象, ...}
        self.along_lane_frame_dict = {}

    def add_traces(self, trace_ls):
        """
        trace_ls是一组轨迹数据，格式为[[xywhθ格式的坐标, frame_index], ...]
        """
        # 1. 更新车辆长宽
        # 计算增量轨迹中车辆长宽均值
        length_ls = [trace[0][3] for trace in trace_ls]
        width_ls = [trace[0][2] for trace in trace_ls]
        avg_length = sum(length_ls) / len(length_ls)
        avg_width = sum(width_ls) / len(width_ls)

        if self.length is not None:
            # 求加权平均
            rate_of_add = len(trace_ls) / (len(trace_ls) + len(self.trace_ls))
            avg_length = avg_length * rate_of_add + self.length * (1 - rate_of_add)
            avg_width = avg_width * rate_of_add + self.width * (1 - rate_of_add)
        self.length = avg_length
        self.width = avg_width

        # 2. 将增量轨迹添加到轨迹列表中
        self.trace_ls.extend(trace_ls)

    def add_car_base_along_lane(self, frame_index, pos, angle, speed):
        """
        添加一帧的车辆基本信息
        pos, angle, speed 分别是道路行驶方向的车辆位置、角度、速度
        """
        self.along_lane_frame_dict[frame_index] = CarAlongLane(pos, angle, speed)

    def get_traffic_incidents(self, frame_index):
        """
        获取基于车辆和周围车辆关系可以判断的指定帧车辆匹配的交通事件列表
        返回Incident枚举类列表
        """
        traffic_incidents = []
        if frame_index not in self.along_lane_frame_dict:
            logger.error(f"Error: 还未提取第{frame_index}帧的道路行驶坐标系下的车辆数据")
            return traffic_incidents

        # 获取与周围车辆有关的交通事件
        traffic_incidents.append(self.__get_incident_with_nearby(frame_index))
        # 获取与道路行驶方向有关的交通事件
        traffic_incidents.append(self.__get_incidents_with_lane(frame_index))

        return traffic_incidents

    def __get_incident_with_nearby(self, frame_index):
        """
        获取指定帧所有与周围车辆有关的交通事件列表
        返回Incident枚举类列表
        """
        # todo:获取与周围车辆有关的交通事件
        traffic_incidents = []
        # 判断车辆与周围车辆的距离是否满足安全距离
        car_along_lane = self.along_lane_frame_dict[frame_index]
        if car_along_lane.get_gap_to_pre_car() < 10:
            traffic_incidents.append(Incident.TOO_CLOSE_PRE)

        return traffic_incidents

    def __get_incidents_with_lane(self, frame_index):
        """
        获取指定帧所有与道路行驶方向有关的交通事件列表
        返回Incident枚举类列表
        """
        traffic_incidents = []
        if self.along_lane_frame_dict[frame_index].is_retrograde():
            traffic_incidents.append(Incident.BACK_UP)

        return traffic_incidents

    def get_box_on_frame(self, extract_frame):
        """
        获取指定帧的车辆位置，这是绝对坐标系下的位置
        """
        trace_index = self.__get_trace_index_by_frame(extract_frame)
        if trace_index is None:
            return None

        return self.trace_ls[trace_index][0]

    def get_speed_on_frame(self, extract_frame, time_span=10):
        """
        获取指定帧的车辆速度矢量，这是绝对坐标系下的速度矢量
        """
        start_index = self.__get_trace_index_by_frame(extract_frame)
        if start_index is None:
            return None

        # 车辆行驶时间太短，无法计算速度
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

    def is_extract_frame_valid(self, extract_frame):
        """
        判断提取帧是否有效
        """
        if extract_frame < 0 or self.__get_max_frame() < extract_frame:
            return False
        return True

    def __get_trace_index_by_frame(self, frame_index):
        """
        通过帧数获取轨迹数据的索引，因为轨迹数据是按照帧数递增的，所以可以直接通过帧数获取索引
        """
        max_frame = self.__get_max_frame()
        start_index = max_frame - frame_index

        if start_index < 0 or start_index >= len(self.trace_ls):
            return None

        return start_index

    def __get_max_frame(self):
        """
        获取轨迹中最大的帧数
        """
        if len(self.trace_ls) == 0:
            return -1
        return self.trace_ls[-1][1]

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

class CarAlongLane:
    """
    存储一帧中在道路行驶方向坐标下的车辆的位置和周围车辆等信息信息
    """
    def __init__(self, pos, angle, speed):
        # 车辆位置，在道路行驶方向坐标系下
        self.pos = pos
        # 车辆速度，在道路行驶方向坐标系下
        self.speed = speed
        # 车辆角度
        self.angle = angle
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

    def is_retrograde(self):
        """
        判断车辆是否逆行，也就是车辆行驶方向和道路行驶方向角度相差角度是否大于阈值
        """
        return math.fabs(self.angle)  > 150

    def get_gap_to_pre_car(self):
        """
        获取与前车的距离
        """
        if self.pre_car is None:
            return float('inf')
        return self.pre_car.pos[0] - self.pos[0], self.pre_car.pos[1] - self.pos[1]

    def get_gap_to_behind_car(self):
        """
        获取与后车的距离
        """
        if self.behind_car is None:
            return float('inf')
        return self.behind_car.pos[0] - self.pos[0], self.behind_car.pos[1] - self.pos[1]

    def get_gap_to_left_pre_car(self):
        """
        获取与左前车的距离
        """
        if self.left_pre_car is None:
            return float('inf')
        return self.left_pre_car.pos[0] - self.pos[0], self.left_pre_car.pos[1] - self.pos[1]

    def get_gap_to_right_pre_car(self):
        """
        获取与右前车的距离
        """
        if self.right_pre_car is None:
            return float('inf')
        return self.right_pre_car.pos[0] - self.pos[0], self.right_pre_car.pos[1] - self.pos[1]

    def get_gap_to_left_behind_car(self):
        """
        获取与左后车的距离
        """
        if self.left_behind_car is None:
            return float('inf')
        return self.left_behind_car.pos[0] - self.pos[0], self.left_behind_car.pos[1] - self.pos[1]

    def get_gap_to_right_behind_car(self):
        """
        获取与右后车的距离
        """
        if self.right_behind_car is None:
            return float('inf')
        return self.right_behind_car.pos[0] - self.pos[0], self.right_behind_car.pos[1] - self.pos[1]