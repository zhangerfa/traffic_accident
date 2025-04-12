"""
提供处理和展示视频中指定帧的接口
"""
import datetime
import logging

import cv2

from incident.car import Car
from incident.obb_tracker import ObbTracker
from incident.incident_processor import extract_speed_ls_from_cars, cluster_speed_vector
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
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_trace(self, start_frame, end_frame, motion_matrix_ls=None, camera_matrix=None):
        # todo: 当target_frame ＜ cur_frame_index时，应该只保留前target_frame帧的轨迹，需要两个容器来存储轨迹，一个用于计算，一个用于缓存，函数总是返回用于计算的容器
        """
        从视频中提取[start_frame, end_frame]帧索引范围内的轨迹数据，返回值:{car_id: List[Box], ...}
        需要注意的是：为了保证轨迹的连续，连续调用该接口时入参的帧索引范围也应该连续，如果两次调用该接口所获取的轨迹的帧范围不连续，那么轨迹也会出现不连续，可能造成同一辆车的轨迹被识别为两辆车的轨迹的情况


        motion_matrix_ls, camera_matrix: 无人机运动矩阵列表，相机矩阵，用于后续投影
        """
        if target_frame is None:
            target_frame = self.frame_count

        # 从self.__trace_frame_count + 1帧开始提取轨迹
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
                    car_dict[obj_id] = Car(obj_id, class_id, 1 / self.fps)

                car_dict[obj_id].trace_ls.append([box, cur_frame_index])

            cur_frame_index += 1
            if cur_frame_index >= target_frame:
                break

    ##################### 轨迹展示方法 #####################
    def show_speed_cluster(self, extract_frame, time_span=10):
        """
        查看车辆速度矢量的聚类情况
        """
        speed_ls = extract_speed_ls_from_cars(self.lane_space.get_cars_on_frame(extract_frame),
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
                cars_on_cur_frame.append(self.lane_space.car_dict[obj_id])
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
        self.get_trace(target_from)
        draw_trace_on_frame(self.lane_space.get_cars_on_frame(show_frame_index), frame)

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