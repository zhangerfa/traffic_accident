"""
提供视频的识别和追踪的工具函数
"""
import csv
import datetime

import cv2

from obb_tracker import logger, ObbTracker
from utils import draw_utils
from utils.predict_utils import initialize_video_capture

def extract_trace_to_csv(yolo_weight_path, video_path, output, frame_count=None, conf=0.5, class_id_id=None):
    """
    从视频中提取轨迹并保存到csv文件
    frame_count: 提取的帧数，如果为None则提取整个视频
    """
    trace_dict = extract_trace(yolo_weight_path, video_path, output, frame_count, conf, class_id_id)
    fieldnames = ['obj_id', 'trace_time', 'class_id', 'x', 'y', 'w', 'h', 'angle']

    with open(output_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 写入列名（表头）
        writer.writeheader()
        # 写入数据
        for obj_id, trace_ls in trace_dict.items():
            box = trace_ls[0]
            row = {
                'obj_id': obj_id,
                'trace_time': trace_ls[0],
                'class_id': trace_ls[1],
                'x': box[0],
                'y': box[1],
                'w': box[2],
                'h': box[3],
                'angle': box[4]
            }
            writer.writerow(row)
    logger.info(f"轨迹已保存到{output}")


def extract_trace(yolo_weight_path, video_path, frame_count=None, conf=0.5, class_id_id=None):
    obb_tracker, cap, fps, frame_width, frame_height = __init(yolo_weight_path, video_path, conf)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cur_frame = 0
    gap = 1
    notify_frame_count = total_frames // gap
    notify_times = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 预测并追踪
        obb_tracker.predict_and_track_frame(frame, cur_frame, class_id_id)

        cur_frame += 1
        if cur_frame == notify_frame_count:
            logger.info(f"已处理{notify_times * gap}%")
            notify_times += 1
        if frame_count is not None and cur_frame >= frame_count:
            break
    return obb_tracker.trace_dict


def predict_and_track_video(yolo_weight_path, video_path, output=None, conf=0.5, class_id_id=None):
    obb_tracker, cap, fps, frame_width, frame_height = __init(yolo_weight_path, video_path, conf)
    # 输出视频
    if output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))
    # 展示时分类颜色
    colors = draw_utils.gene_colors(obb_tracker.__class_names)

    while True:
        start = datetime.datetime.now()
        ret, frame = cap.read()
        if not ret:
            break
        # 预测并追踪
        objs = obb_tracker.predict_and_track_frame(frame, conf, class_id_id)
        # 展示：画框，并添加文本：obj_id - class_name，也就是这是第 obj_id 个 class_name
        for obj in objs:
            box, obj_id, class_id = obj
            text = f"{obj_id} - {obb_tracker.__class_names[class_id]}"
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

    if cap is not None:
        cap.release()
    if output is not None:
        writer.release()
    cv2.destroyAllWindows()

def __init(yolo_weight_path, video_path, conf=0.5):
    # 加载模型和视频
    obb_tracker = ObbTracker(yolo_weight_path, conf)
    cap = initialize_video_capture(video_path)
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    return obb_tracker, cap, fps, frame_width, frame_height

if __name__ == "__main__":
    output_path = r"D:\download\output.csv"
    yolo_weight_path = '../weights/yolo11x-obb.pt'
    video_path = r"E:\data\self\video\DJI_0804.MP4"

    extract_trace_to_csv(yolo_weight_path, video_path, output_path, 100, conf=0.8)