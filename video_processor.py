"""
视频预测、追踪、提取轨迹并提供基础展示功能
"""
import datetime
import logging

import cv2

from obb_tracker import ObbTracker
from utils.draw_utils import draw_box, gene_colors, draw_trace_on_frame
from utils.predict_utils import predict_and_show_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, yolo_weight_path, video_path, conf=0.6):
        # 加载模型和视频
        self.obb_tracker = ObbTracker(yolo_weight_path, conf)
        self.cap = self.__initialize_video_capture(video_path)
        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        # 缓存提取的轨迹
        self.__trace_dict = {}

    def show_trace(self):
        """
        提取轨迹并展示
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            draw_trace_on_frame(self.obb_tracker.trace_dict, frame)
            cv2.imshow("Vehicle Trace", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def extract_trace(self, frame_count=None, class_id_id=None):
        """
        从视频中提取轨迹，如果已经提取过则直接返回
        """
        if self.__trace_dict != {}:
            return self.__trace_dict

        # 获取视频的总帧数
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cur_frame = 0
        gap = 1
        notify_frame_count = total_frames // gap
        notify_times = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 预测并追踪
            self.obb_tracker.predict_and_track_frame(frame, cur_frame, class_id_id)

            cur_frame += 1
            if cur_frame == notify_frame_count:
                logger.info(f"已处理{notify_times * gap}%")
                notify_times += 1
            if frame_count is not None and cur_frame >= frame_count:
                break
        return self.obb_tracker.trace_dict

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

    def get_frame(self, frame_id):
        """
        获取指定帧
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            logger.error(f"Error: 未找到第{frame_id}帧")
            return None
        return frame

    def __initialize_video_capture(self, video_input):
        if video_input.isdigit():
            video_input = int(video_input)
            cap = cv2.VideoCapture(video_input)
        else:
            cap = cv2.VideoCapture(video_input)

        if not cap.isOpened():
            logger.error("Error: 视频不存在")
            raise ValueError("Unable to open video source")

        return cap