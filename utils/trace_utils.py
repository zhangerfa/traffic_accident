import logging

import numpy as np
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cal_lane_from_trace(cars, start_frame, time_span=10):
    """
    从轨迹中计算车道，车道包含车道方向和宽度衡量的范围
    cars: Car实例对象列表
    start_frame: 起始帧
    time_span: 计算速度矢量时的时间跨度，单位为帧数
    """
    lane_direction = {}
    # 获取指定帧的所有车辆速度矢量
    speed_vector_ls = extract_speed_ls_from_cars(cars, start_frame, time_span)
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


def __norm_speed_vector(speed_ls):
    """
    归一化速度矢量，归一化后的速度矢量就只描述速度方向
    """
    # 计算速度矢量的角度，以弧度表示
    angles = [np.arctan2(v, u) for u, v in speed_ls]
    angles = np.array(angles).reshape(-1, 1)

    return angles