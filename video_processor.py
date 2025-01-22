"""
集成视频识别、追踪模型，轨迹处理对象
"""
import datetime
import logging

import cv2

from obb_tracker import ObbTracker
from utils.draw_utils import draw_box, gene_colors, draw_trace_on_frame, draw_speed_cluster
from utils.predict_utils import predict_and_show_frame
from utils.trace_utils import cal_lane_from_trace, extract_trace_to_csv, load_trace_from_csv, \
    extract_speed_vector_from_trace, cluster_speed_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, yolo_weight_path, video_path, conf=0.6):
        # 加载模型和视频
        self.obb_tracker = ObbTracker(yolo_weight_path, conf)
        self.cap = self.initialize_video_capture(video_path)
        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        # 从视频中提取的轨迹数据：{obj_id: [[xywhθ格式的检测框坐标, frame_index, cls_id], ...], ...}
        self.trace_dict = {}
        # 当前轨迹已经提取的帧数范围：[0, trace_frame_count]，也就是说当前已经提取了前trace_frame_count帧的轨迹
        self.trace_frame_count = -1

    def cal_lane_direction(self, frame_index, time_span=10, class_id_id=None):
        """
        从轨迹中计算第frame_index帧中的车道行驶方向
        time_span: 估算车道行驶方向时的计算车辆速度矢量时的时间跨度，单位为帧数
        """
        # 提取0~frame_index帧的轨迹
        self.__extract_trace(frame_index, class_id_id)
        return cal_lane_from_trace(self.trace_dict, frame_index, time_span)

    def extract_trace_to_csv(self, output_path, frame_count=None, class_id_id=None):
        """
        从视频中提取轨迹并保存为csv文件
        """
        self.__extract_trace(frame_count, class_id_id)
        extract_trace_to_csv(self.trace_dict, output_path, self.trace_frame_count)

    def load_trace_from_csv(self, csv_file_path):
        # 从csv文件中加载轨迹数据，其中csv文件中的元数据为轨迹提取范围：trace_frame_count
        self.trace_dict, self.trace_frame_count = load_trace_from_csv(csv_file_path)

    def show_trace(self):
        """
        无人机静止情况下提取轨迹并展示：将同一辆车的轨迹用一系列点表示（如果无人机运动图片坐标系就变了，需要坐标系转换）
        """
        cur_frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 预测并追踪
            objs = self.__update_trace(frame, cur_frame_index)
            # 只保留当前帧中出现车辆的轨迹
            cur_trace_dict = {}
            for obj in objs:
                box, obj_id, class_id = obj
                cur_trace_dict[obj_id] = self.trace_dict[obj_id]
            draw_trace_on_frame(cur_trace_dict, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def show_trace_on_frame(self, frame_index, frame_count=None):
        """
        将轨迹数据画到指定帧上并展示，如果当前视频中没有提取过轨迹则先提取
        frame_index: 帧索引
        """
        frame = self.__get_frame(frame_index)
        self.__extract_trace(frame_count)
        draw_trace_on_frame(self.trace_dict, frame)

    def predict_and_track(self, output=None, class_id_id=None):
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
            objs = self.obb_tracker.predict_and_track_frame(frame, class_id_id)
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

    def __extract_trace(self, frame_count=0, class_id_id=None):
        """
        从视频中提取轨迹，如果已经提取过则直接返回
        frame_count: 提取轨迹的帧数，如frame_count=200，就是提取前200帧的轨迹，如果frame_count=None，则提取整个视频的轨迹
        """
        if frame_count is None:
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.trace_frame_count >= frame_count:
            return

        # 从trace_frame_count + 1帧开始提取轨迹
        cur_frame_index = self.trace_frame_count + 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_index)

        logger.info(f"正在获取{self.trace_frame_count + 1} ~ "
                    f"{min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_count)} 帧的轨迹数据")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 预测并追踪当前帧，更新trace_dict
            self.__update_trace(frame, cur_frame_index, class_id_id)

            cur_frame_index += 1
            if frame_count is not None and cur_frame_index >= frame_count:
                break

        self.trace_frame_count = cur_frame_index - 1

    def __update_trace(self, frame, cur_frame, class_id_id=None):
        """
        预测并追踪当前帧，更新trace_dict，注意：传入帧必须为当前轨迹的trace_frame_count + 1帧
        """
        objs = self.obb_tracker.predict_and_track_frame(frame, class_id_id)
        for obj in objs:
            box, obj_id, class_id = obj
            if obj_id not in self.trace_dict:
                self.trace_dict[obj_id] = []

            self.trace_dict[obj_id].append([box, cur_frame, class_id])

        return objs

    @classmethod
    def initialize_video_capture(cls, video_input):
        if video_input.isdigit():
            video_input = int(video_input)
            cap = cv2.VideoCapture(video_input)
        else:
            cap = cv2.VideoCapture(video_input)

        if not cap.isOpened():
            logger.error("Error: 视频不存在")
            raise ValueError("Unable to open video source")

        return cap

    @classmethod
    def show_speed_cluster(cls, trace_dict, extract_frame, time_span=10):
        """
        查看车辆速度矢量的聚类情况
        """
        speed_ls = extract_speed_vector_from_trace(trace_dict, extract_frame, time_span)
        cluster_dict = cluster_speed_vector(speed_ls)

        draw_speed_cluster(cluster_dict)
