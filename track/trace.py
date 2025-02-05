"""
轨迹数据及轨迹处理逻辑
"""
import csv
import logging

import numpy as np
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trace:
    """
    轨迹数据及轨迹处理逻辑
    """
    def __init__(self):
        # 车辆数据：{car_id: Car实例对象}
        self.car_dict = {}
        # 当前轨迹已经提取的帧数范围：[0, trace_frame_count]，也就是说当前已经提取了前trace_frame_count帧的轨迹
        self.trace_frame_count = -1

    def get_cars(self):
        """
        获取所有车辆对象
        """
        return list(self.car_dict.values())

    def cal_lane_direction(self, frame_index, time_span=10):
        """
        从轨迹中计算第frame_index帧中的车道行驶方向
        time_span: 估算车道行驶方向时的计算车辆速度矢量时的时间跨度，单位为帧数
        """
        if frame_index > self.trace_frame_count:
            raise ValueError(f"Error: 当前轨迹中不包含第{frame_index}帧的轨迹数据，请在调用Trace接口前更新轨迹数据")

        cars = self.get_cars()
        lane_direction = {}
        # 获取指定帧的所有车辆速度矢量
        speed_vector_ls = extract_speed_ls_from_cars(cars, frame_index, time_span)
        if len(speed_vector_ls) == 0:
            return lane_direction
        # 对车辆速度矢量进行聚类
        cluster_dict = cluster_speed_vector(speed_vector_ls)
        # 使用统计方法剔除明显偏离车道方向的速度矢量
        filterSpeed(cluster_dict)
        # 每组车辆速度矢量的矢量和的方向作为车道方向
        for cluster_id, speed_vector_ls in cluster_dict.items():
            speed_vector_ls = np.array(speed_vector_ls)
            speed_vector_sum = np.sum(speed_vector_ls, axis=0)
            lane_direction[cluster_id] = np.degrees(np.arctan2(speed_vector_sum[1], speed_vector_sum[0]))

        return lane_direction

    def to_csv(self, output):
        """
        将轨迹数据保存到指定的csv文件
        """
        cars = list(self.car_dict.values())

        fieldnames = ['obj_id', 'trace_time', 'class_id', 'x', 'y', 'w', 'h', 'angle']

        with open(output, "w", newline='') as f:
            # 1. 插入元数据
            f.write(f'# frame_count: {self.trace_frame_count}\n')
            # 2. 插入轨迹数据
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 写入列名（表头）
            writer.writeheader()
            # 写入数据
            for car in cars:
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

    def load_trace_from_csv(self, csv_file_path):
        """
        从csv文件中加载轨迹数据
        其中csv文件中的元数据为已提取轨迹的帧范围：trace_frame_count，如frame_count=200表示已提取前200帧的轨迹
        """
        with open(csv_file_path, 'r') as f:
            # 读取元数据
            metadata = f.readline().strip()
            if metadata.startswith("# frame_count:"):
                self.trace_frame_count = int(metadata.split(":")[-1])
            else:
                logger.error("Error: 未找到轨迹数据的元数据{trace_frame_count}")
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = int(row['obj_id'])
                trace_time = int(row['trace_time'])
                class_id = row['class_id']
                box = [float(row['x']), float(row['y']), float(row['w']), float(row['h']), float(row['angle'])]

                if obj_id not in self.car_dict:
                    self.car_dict[obj_id] = Car(obj_id, class_id)
                self.car_dict[obj_id].trace_ls.append([box, trace_time])
        logger.info(f"轨迹已从{csv_file_path}加载")

    def get_start_update_frame(self):
        """
        获取需要更新轨迹数据的起始帧
        """
        return self.trace_frame_count + 1

    def add_trace(self, car_dict, trace_frame_count):
        """
        添加轨迹数据
        """
        for obj_id, trace_ls in car_dict.items():
            if obj_id not in self.car_dict:
                self.car_dict[obj_id] = Car(obj_id, trace_ls[0][2])
            self.car_dict[obj_id].trace_ls.extend(trace_ls)
        self.trace_frame_count = trace_frame_count

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

    # 聚类的输入为归一化的速度矢量，这样可以使得速度大小不影响聚类结果，只考虑速度方向
    speed_ls_norm = np.array(speed_vector_ls)
    speed_ls_norm = speed_ls_norm / np.linalg.norm(speed_ls_norm, axis=1)[:, np.newaxis]
    labels = cluster.fit_predict(speed_ls_norm)

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
        angles = __norm_speed_vector(speed_vector_ls)
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

def __norm_speed_vector(speed_ls):
    """
    归一化速度矢量，归一化后的速度矢量就只描述速度方向
    """
    # 计算速度矢量的角度，以弧度表示
    angles = [np.arctan2(v, u) for u, v in speed_ls]
    angles = np.array(angles).reshape(-1, 1)

    return angles