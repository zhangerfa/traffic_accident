"""
提供视频的识别和追踪的工具函数
"""

import datetime

import cv2

from obb_tracker import logger, ObbTracker
from utils import draw_utils
from utils.predict_utils import initialize_video_capture


def predict_and_track_video(yolo_weight_path, video_path, output=None, conf=0.5, bluf_id=None, class_id_id=None):
    try:
        # 加载模型和视频
        obb_tracker = ObbTracker(yolo_weight_path)
        cap = initialize_video_capture(video_path)
        # 获取视频信息
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 输出视频
        if output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))
        # 展示时分类颜色
        colors = draw_utils.gene_colors(obb_tracker.class_names)

        while True:
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break
            # 预测并追踪
            objs = obb_tracker.predict_and_track(frame, conf, class_id_id)
            # 展示：画框，并添加文本：obj_id - class_name，也就是这是第 obj_id 个 class_name
            for obj in objs:
                box, obj_id, class_id = obj
                text = f"{obj_id} - {obb_tracker.class_names[class_id]}"
                frame = draw_utils.draw_box(frame, box, text, colors[class_id])

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

    except Exception as e:
        logger.exception(e)
    finally:
        if cap is not None:
            cap.release()
        if output is not None:
            writer.release()
        cv2.destroyAllWindows()
