"""
轨迹数据及轨迹处理逻辑
"""
import csv
import logging

import numpy as np
from sklearn.cluster import DBSCAN

from incident.Incident import Incident
from incident.car import Car
from incident.along_lane_car import AlongLaneCar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentProcessor:
    """
    轨迹数据及轨迹处理逻辑
    """
    def __init__(self, video_processor):
        # 车辆数据：{car_id: Car实例对象}
        self.car_dict = {}
        # 道路行驶方向
        self.lane_direction_dict = {}
        # 视频处理器
        self.video_processor = video_processor
        # 当前轨迹已经提取的帧数范围：[0, __trace_frame_count]，也就是说当前已经提取了前trace_frame_count帧的轨迹
        self.__trace_frame_count = -1

    def get_traffic_incidents_online(self, pic, motion_matrix, camera_matrix):
        """
        通过网络连续传入图片，将图片数据添加到消息队列中，消息队列内部会获取交通事件并将交通事件存储到数据库中
        需要注意的是：调用该接口时连续传入的图片应该是视频中的连续帧，否则会导致车辆轨迹不连续，从而识别不到交通事件

        pic: 输入图片
        motion_matrix: 无人机运动矩阵
        camera_matrix: 相机矩阵
        """

    def get_traffic_incidents_offline(self, start_frame=None, end_frame=None):
        """
        获取[start_frame, end_frame]帧索引范围内视频中的所有交通事件，如果start_frame为None则提取整个视频的交通事件
        返回格式：{"micro_incidents": {car_id: {Incident: 发生事件的帧数列表}, ...}, "macro_incidents": {Incident: 发生事件的帧数列表},...}
        """
        # todo:
        if start_frame is None:
            start_frame = 0
            end_frame = self.video_processor.frame_count

        fps = self.video_processor.fps

        # 每隔三分之一秒提取一次交通事件
        traffic_incidents = {}
        micro_traffic_incidents = {}
        macro_traffic_incidents = {}
        traffic_incidents["micro_incidents"] = micro_traffic_incidents
        traffic_incidents["macro_incidents"] = macro_traffic_incidents
        for target_frame in range(start_frame, end_frame, fps // 3):
            # 提取微观交通事件
            traffic_incidents_on_frame = self.__get_micro_traffic_incidents_on_frame(target_frame)
            if len(traffic_incidents_on_frame) > 0:
                for car_id, incidents in traffic_incidents_on_frame.items():
                    if car_id not in micro_traffic_incidents:
                        micro_traffic_incidents[car_id] = {}
                    for incident in incidents:
                        if incident not in micro_traffic_incidents[car_id]:
                            micro_traffic_incidents[car_id][incident] = []
                        micro_traffic_incidents[car_id][incident].append(target_frame)
            # 提取宏观交通事件
            macro_traffic_incidents = self.__get_macro_traffic_incidents_on_frame(target_frame)
            if len(macro_traffic_incidents) > 0:
                for incident in macro_traffic_incidents:
                    if incident not in macro_traffic_incidents:
                        macro_traffic_incidents[incident] = []
                    macro_traffic_incidents[incident].append(target_frame)

        return traffic_incidents

    def __get_macro_traffic_incidents_on_frame(self, target_frame):
        """
        获取指定帧的宏观交通事件列表，返回格式为：[Incident]
        """
        macro_incidents = []
        # 统计前车距超过5m的车辆
        car_ls = self.__gene_along_lane_cars(target_frame)
        jam_car_ls = [car for car in car_ls if car.get_distance_to_target(car.pre_car) > 5]
        if len(jam_car_ls) / len(car_ls) >= 0.6:
            macro_incidents.append(Incident.JAM)
        return macro_incidents

    def __get_micro_traffic_incidents_on_frame(self, target_frame):
        """
        获取指定帧所有微观交通事件列表，返回格式为：{car_id:Incident列表}
        """
        # todo: 获取交通事件时只提取需要的帧范围的轨迹
        self.__update_trace(target_frame)
        cars = self.get_cars_on_frame(target_frame)
        traffic_incident_dict = {}
        # 1. 遍历所有车辆提取基于车辆自身的交通事件列表
        for car in cars:
            cur_incidents = car.__get_micro_traffic_incidents_on_frame(target_frame)
            if len(cur_incidents) > 0:
                traffic_incident_dict[car.id] = cur_incidents
        # 提取当前帧车辆的周围关系
        along_lane_cars =  self.__gene_along_lane_cars(target_frame)
        for along_lane_car in along_lane_cars:
            # 2. 获取基于车辆周围车辆关系的交通事件列表
            cur_incidents = along_lane_car.get_traffic_incidents()
            # 3. 提取基于道路行驶规则的交通事件列表
            if along_lane_car.angle > 150:
                cur_incidents.append(Incident.OVER_SPEED)
            # 合并当前车所有类型的交通事件
            if len(cur_incidents) > 0:
                if along_lane_car.id not in traffic_incident_dict:
                    traffic_incident_dict[along_lane_car.id] = []
                traffic_incident_dict[along_lane_car.id].extend(cur_incidents)

        return traffic_incident_dict

    # todo: lane_space中两个变量标记已经提取轨迹的范围
    def __update_trace(self, start_frame, end_frame, motion_matrix=None, camera_matrix=None):
        trace = self.video_processor.get_trace(self.__trace_frame_count + 1, end_frame)
        self.__trace_frame_count = {}

    def __update_trace_by_pic(self, pic):
        """
        通过网络连续传入图片，将图片数据添加到消息队列中，消息队列内部会获取交通事件并将交通事件存储到数据库中
        """

    def __gene_along_lane_cars(self, target_frame):
        """
        提取指针帧的沿着道路行驶方向的车辆数据
        """
        # 1. 提取目标帧中的所有车辆，并将其坐标投影到道路行驶坐标系
        along_lane_cars_on_target_frame = []
        for car in self.get_cars_on_frame(target_frame):
            # 将车辆的坐标、速度转换到道路行驶坐标系
            box = car.get_box_on_frame(target_frame)
            speed = car.get_speed_on_frame(target_frame)
            if box or speed is None:
                continue
            pos, angle = self.__trans_to_lane_direction(target_frame, [box[0], box[1]], box[4])
            speed, _ = self.__trans_to_lane_direction(target_frame, speed)

            along_lane_car = AlongLaneCar(pos, speed, angle)
            along_lane_cars_on_target_frame.append(along_lane_car)
        # 2. 将车辆按道路行驶方向的远近排序

        # todo:3. 提取车辆的周围车辆信息
        for along_lane_car in along_lane_cars_on_target_frame:
            pass

        return along_lane_cars_on_target_frame

    def __trans_to_lane_direction(self, target_frame, pos=None, angle=None):
        """
        将向量、角度转换到道路行驶方向坐标系
        """
        x_new = None
        y_new = None
        angle_new = None

        lane_direction = self.cal_lane_direction(target_frame)

        # todo：车辆和道路行驶方向配对问题（道路行驶方向边界问题）
        if len(lane_direction) == 0:
            return [-1, -1], -1
        lane_angle = lane_direction[0]
        if pos is not None:
            x, y = pos

            # 计算旋转矩阵
            cos_theta = np.cos(lane_angle)
            sin_theta = np.sin(lane_angle)

            x_new = cos_theta * x - sin_theta * y
            y_new = sin_theta * x + cos_theta * y

        if angle is not None:
            angle_new = angle - lane_angle

        return [x_new, y_new], angle_new

    def get_cars_on_frame(self, target_frame=None):
        """
        获取指定帧上的所有车辆对象，如果不传入target_frame则获取所有车辆
        """
        if target_frame is not None:
            cars = []
            for car in self.car_dict.values():
                if car.is_on_frame(target_frame):
                    cars.append(car)
            return cars
        return list(self.car_dict.values())

    def cal_lane_direction(self, target_frame, time_span=10):
        """
        从轨迹中计算第target_frame帧中的车道行驶方向，单位为弧度
        time_span: 估算车道行驶方向时的计算车辆速度矢量时的时间跨度，单位为帧数
        """
        if target_frame in self.lane_direction_dict:
            return self.lane_direction_dict[target_frame]
        if target_frame > self.__trace_frame_count:
            raise ValueError(f"Error: 当前轨迹中不包含第{target_frame}帧的轨迹数据，请在调用Trace接口前更新轨迹数据")

        cars = self.get_cars_on_frame(target_frame)
        lane_direction = {}
        # 获取指定帧的所有车辆速度矢量
        speed_vector_ls = extract_speed_ls_from_cars(cars, target_frame, time_span)
        if len(speed_vector_ls) == 0:
            return lane_direction
        # 对速度归一化，这样可以使得速度大小不影响聚类结果，只考虑速度方向
        speed_ls_norm = np.array(speed_vector_ls)
        speed_ls_norm = speed_ls_norm / np.linalg.norm(speed_ls_norm, axis=1)[:, np.newaxis]
        # 对车辆速度矢量进行聚类
        cluster_dict = cluster_speed_vector(speed_vector_ls)
        # 使用统计方法剔除明显偏离车道方向的速度矢量
        filterSpeed(cluster_dict)
        # 每组车辆速度矢量的矢量和的方向作为车道方向
        for cluster_id, speed_vector_ls in cluster_dict.items():
            speed_vector_ls = np.array(speed_vector_ls)
            speed_vector_sum = np.sum(speed_vector_ls, axis=0)
            lane_direction[cluster_id] = np.degrees(np.arctan2(speed_vector_sum[1], speed_vector_sum[0]))

        self.lane_direction_dict[target_frame] = lane_direction
        return lane_direction

    def save_trace_to_csv(self, output, start_frame=0, end_frame=None):
        """
        将帧索引[start_frame, end_frame]范围内的迹数据保存到指定的csv文件
        """
        self.video_processor.update_trace(target_frame)

        cars = list(self.car_dict.values())

        fieldnames = ['obj_id', 'trace_time', 'class_id', 'x', 'y', 'w', 'h', 'angle']

        with open(output, "w", newline='') as f:
            # 1. 插入元数据
            f.write(f'# frame_count: {self.__trace_frame_count}\n')
            # 2. 插入轨迹数据
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 写入列名（表头）
            writer.writeheader()
            # 写入每辆车的轨迹
            for car in cars:
                # todo car做个接口
                for trace in car.trace_ls:
                    box = trace[0]
                    row = {
                        'obj_id': car.id,
                        'trace_time': trace[1],
                        'class_id': car.class_id,
                        'x': box[0],
                        'y': box[1],
                        'w': box[2],
                        'h': box[3],
                        'angle': box[4]
                    }
                    writer.writerow(row)
            logger.info(f"轨迹已保存到{output}")

    def load_trace_from_csv(self, csv_file_path, start_frame=0, end_frame=None):
        """
        从csv文件中加帧索引[start_frame, end_frame]范围内的迹数据
        其中csv文件中的元数据为已提取轨迹的帧范围：__trace_frame_count，如frame_count=200表示已提取前200帧的轨迹
        """
        with open(csv_file_path, 'r') as f:
            # 读取元数据
            metadata = f.readline().strip()
            if metadata.startswith("# frame_count:"):
                self.__trace_frame_count = int(metadata.split(":")[-1])
            else:
                logger.error("Error: 未找到轨迹数据的元数据{__trace_frame_count}")
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = int(row['obj_id'])
                trace_time = int(row['trace_time'])
                class_id = row['class_id']
                box = [float(row['x']), float(row['y']), float(row['w']), float(row['h']), float(row['angle'])]

                if obj_id not in self.car_dict:
                    self.car_dict[obj_id] = Car(obj_id, class_id, 1 / self.video_processor.fps)
                # todo car做个接口
                self.car_dict[obj_id].trace_ls.append([box, trace_time])
        logger.info(f"轨迹已从{csv_file_path}加载")

def extract_speed_ls_from_cars(cars, extract_frame, time_span=10):
    """
    从车辆轨迹中提取车辆速度矢量：(start_frame - time_span)帧到start_frame帧的速度矢量
    cars: Car实例对象列表
    extract_frame: 提取速度矢量的起始帧
    time_span: 计算速度矢量时的时间跨度，单位为帧数，从start_frame-time_span帧到start_frame帧
    """
    speed_vector_ls = []

    for car in cars:
        speed_vector = car.get_speed_on_frame(extract_frame, time_span)
        if speed_vector is None:
            continue
        # 剔除速度矢量极小的点
        if np.linalg.norm(speed_vector) < 5:
            continue
        speed_vector_ls.append(speed_vector)

    return speed_vector_ls

def cluster_speed_vector(speed_vector_ls):
    """
    对车辆的速度矢量进行聚类，返回聚类结果
    """
    # 使用DBSCAN聚类，eps代表两点距离的阈值，min_samples是最小样本数
    cluster = DBSCAN(eps=1, min_samples=2)
    labels = cluster.fit_predict(speed_vector_ls)

    # 将每个类别的行驶方向聚合为列表
    cluster_dict = {}
    for i in range(len(labels)):
        # -1代表噪声点
        if labels[i] == -1:
            continue
        if labels[i] not in cluster_dict:
            cluster_dict[labels[i]] = []
        cluster_dict[labels[i]].append(speed_vector_ls[i])

    return cluster_dict

def filterSpeed(cluster_dict):
    """
    使用统计方法剔除明显偏离车道方向的速度矢量
    """
    for cluster_id, speed_vector_ls in cluster_dict.items():
        # 对速度矢量归一化来描述速度方向
        angles = Car.get_speed_angle(speed_vector_ls)
        # 对速度方向求和后归一化来描述平均速度方向
        angle_avg = np.mean(angles)
        # 计算每个速度矢量与平均速度方向的夹角
        angle_diff = np.abs(angles - angle_avg)
        # 计算夹角的标准差
        angle_diff_std = np.std(angle_diff)
        # 剔除夹角大于2倍标准差的速度矢量
        cluster_dict[cluster_id] = [speed_vector
                                    for speed_vector, diff in zip(speed_vector_ls, angle_diff)
                                    if diff < 2 * angle_diff_std]
    # 剔除空的聚类
    for cluster_id in list(cluster_dict.keys()):
        if len(cluster_dict[cluster_id]) == 0:
            del cluster_dict[cluster_id]