"""
提供处理和展示视频中指定帧的接口
"""
import datetime
import logging

import cv2

from incident.car import Car
from incident.obb_tracker import ObbTracker
from incident.incident_processor import IncidentProcessor, extract_speed_ls_from_cars, cluster_speed_vector
from incident.utils.draw_utils import draw_box, gene_colors, draw_trace_on_frame, draw_speed_cluster
from train.utils.predict_utils import predict_and_show_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, yolo_weight_path, video_path, conf=0.6, specified_class_id=None):
        # 加载模型和视频
        self.obb_tracker = ObbTracker(yolo_weight_path, conf, specified_class_id)
        self.cap = initialize_video_capture(video_path)
        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        # 交通事件处理器
        self.incident_processor = IncidentProcessor()

    ##################### 轨迹数据处理 #####################
    def get_traffic_incidents(self, target_frame=None):
        """
        获取前target_frame帧视频中的所有交通事件，如果target_frame为None则提取整个视频的交通事件
        返回格式：{car_id: {事件名称: 发生事件的帧数列表}, ...}
        """
        if target_frame is None:
            target_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 预热1秒
        if target_frame < self.fps:
            logger.error("Error: 前1秒无法提取交通事件")
            return {}
        self.__update_trace(self.fps)

        # 每隔三分之一秒提取一次交通事件
        traffic_incidents = {}
        for frame_index in range(self.fps + 1, target_frame, self.fps // 3):
            traffic_incidents_on_frame = self.__get_traffic_incidents_on_frame(frame_index)
            if len(traffic_incidents_on_frame) > 0:
                for car_id, incidents in traffic_incidents_on_frame.items():
                    if car_id not in traffic_incidents:
                        traffic_incidents[car_id] = {}
                    for incident in incidents:
                        if incident not in traffic_incidents[car_id]:
                            traffic_incidents[car_id][incident] = []
                        traffic_incidents[car_id][incident].append(frame_index)

        return traffic_incidents

    def __get_traffic_incidents_on_frame(self, frame_index):
        """
        获取指定帧所有交通事件列表，返回格式为：{car_id:交通事件列表}
        """
        self.__update_trace(frame_index)
        return self.incident_processor.get_traffic_incidents(frame_index)

    def cal_lane_direction(self, frame_index):
        """
        从轨迹中计算第frame_index帧中的车道行驶方向
        """
        self.__update_trace(frame_index)
        return self.incident_processor.cal_lane_direction(frame_index)

    def extract_trace_to_csv(self, output, target_frame=None):
        """
        从视频中提取前帧的轨迹并保存到output指定的路径下（需要为csv文件），如果target_frame为None则提取整个视频的轨迹
        """
        if output is None or output.split(".")[-1] != "csv":
            logger.error("Error: 未指定输出路径或输出路径不是csv文件")
            return
        self.__update_trace(target_frame)
        self.incident_processor.to_csv(output)

    def load_trace_from_csv(self, csv_file_path):
        """
        从csv文件中读取轨迹数据
        """
        self.incident_processor.load_trace_from_csv(csv_file_path)

    ##################### 轨迹展示方法 #####################
    def show_speed_cluster(self, extract_frame, time_span=10):
        """
        查看车辆速度矢量的聚类情况
        """
        speed_ls = extract_speed_ls_from_cars(self.incident_processor.get_cars_on_frame(extract_frame),
                                              extract_frame, time_span)
        cluster_dict = cluster_speed_vector(speed_ls)

        draw_speed_cluster(cluster_dict)

    def show_trace(self):
        """
        展示数据：将同一辆车的轨迹用一系列点表示（如果无人机运动图片坐标系就变了，需要坐标系转换）
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 预测并追踪
            objs = self.obb_tracker.predict_and_track_frame(frame)
            # 只保留当前帧中出现车辆的轨迹
            cars_on_cur_frame = []
            for obj in objs:
                box, obj_id, class_id = obj
                cars_on_cur_frame.append(self.incident_processor.car_dict[obj_id])
            draw_trace_on_frame(cars_on_cur_frame, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def show_trace_on_frame(self, show_frame_index, target_from=None):
        """
        将前target_frame帧的轨迹数据画到show_frame_index帧上并展示，如果target_from为None则展示整个视频的轨迹
        """
        frame = self.__get_frame(show_frame_index)
        self.__update_trace(target_from)
        draw_trace_on_frame(self.incident_processor.get_cars_on_frame(show_frame_index), frame)

    ##################### 识别和追踪展示方法 #####################

    def predict_and_track(self, output=None):
        # 输出视频
        if output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, self.fps, (self.frame_width, self.frame_height))
        # 展示时分类颜色
        colors = gene_colors(self.obb_tracker.class_names)

        while True:
            start = datetime.datetime.now()
            ret, frame = self.cap.read()
            if not ret:
                break
            # 预测并追踪
            objs = self.obb_tracker.predict_and_track_frame(frame)
            # 展示：画框，并添加文本：obj_id - class_name，也就是这是第 obj_id 个 class_name
            for obj in objs:
                box, obj_id, class_id = obj
                text = f"{obj_id} - {self.obb_tracker.class_names[class_id]}"
                frame = draw_box(frame, box, text, colors[class_id])

            # 展示帧率
            end = datetime.datetime.now()
            fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            # 输出视频
            if output is not None:
                writer.write(frame)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("YOLOv11 Object tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if self.cap is not None:
            self.cap.release()
        if output is not None:
            writer.release()
        cv2.destroyAllWindows()

    def predict(self):
        """
        预测并展示视频，按任意键展示下一帧
        """
        colors = gene_colors(self.obb_tracker.class_names)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            predict_and_show_frame(self.obb_tracker.yolo, frame, colors)
            # 按q键退出
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
        self.cap.release()

    def __get_frame(self, frame_id):
        """
        获取指定帧
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            logger.error(f"Error: 未找到第{frame_id}帧")
            return None
        return frame

    def __update_trace(self, target_frame):
        """
        从视频中提取前target_frame帧的轨迹数据，如果target_frame为None则提取整个视频的轨迹
        """
        if target_frame is None:
            target_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 从self.trace_frame_count + 1帧开始提取轨迹
        cur_frame_index = self.incident_processor.get_start_update_frame()
        if target_frame <= cur_frame_index:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_index)

        logger.info(f"正在获取{cur_frame_index} ~ {target_frame} 帧的轨迹数据")

        car_dict = {}
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 识别和追踪当前帧
            objs = self.obb_tracker.predict_and_track_frame(frame)
            for obj in objs:
                box, obj_id, class_id = obj
                if obj_id not in car_dict:
                    car_dict[obj_id] = Car(obj_id, class_id)

                car_dict[obj_id].trace_ls.append([box, cur_frame_index])

            cur_frame_index += 1
            if cur_frame_index >= target_frame:
                break

        self.incident_processor.add_trace(car_dict, cur_frame_index - 1)

def initialize_video_capture(video_input):
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        logger.error("Error: 视频不存在")
        raise ValueError("Unable to open video source")

    return cap