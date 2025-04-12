import math

from incident.Incident import Incident
from incident.car import Car


class AlongLaneCar:
    """
    存储一帧中在道路行驶方向坐标下的车辆的基本信息和周围车辆等信息信息
    """
    def __init__(self, id, box, speed):
        self.id = id
        x, y, length, width, angle = box
        # 车辆位置，在道路行驶方向坐标系下
        self.pos = [x, y]
        # 车辆速度，在道路行驶方向坐标系下
        self.speed = speed
        # 车辆角度
        self.angle = angle
        # 车辆长度
        self.length = length
        # 车辆宽度
        self.width = width
        # 前车，类型为CarAlongLane实例对象，下同
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

    def get_traffic_incidents(self):
        """
        获取车辆的基于周围车辆关系的交通事件，返回值为Incident列表
        """
        incidents = []
        # 判断是否逆行/倒车
        if Car.get_speed_angle([self.speed])[0] > math.pi / 2:
            incidents.append(Incident.BACK_UP)

        return incidents

    def get_distance_to_target(self, target_car):
        """
        获取与传入车的横向距离和纵向距离
        """
        if target_car is None:
            return float('inf')
        return target_car.pos[0] - self.pos[0], target_car.pos[1] - self.pos[1]











