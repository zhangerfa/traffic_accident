import cv2
import torch
import datetime
import logging
from collections import defaultdict
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

import os

import utils

# 允许多版本 OpenMP 共存
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define command line flags
flags.DEFINE_string("video", "./data/test_1.mp4", "Path to input video or webcam index (0)")
flags.DEFINE_string("output", "./output/output.mp4", "Path to output video")
flags.DEFINE_string("model_path", "./weights/yolo11x-obb.pt", "Path to model")
flags.DEFINE_float("conf", 0.50, "Confidence threshold")
flags.DEFINE_integer("blur_id", None, "Class ID to apply Gaussian Blur")
flags.DEFINE_integer("class_id", None, "Class ID to track")

FLAGS = flags.FLAGS

def initialize_video_capture(video_input):
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        logger.error("Error: Unable to open video source.")
        raise ValueError("Unable to open video source")
    
    return cap

def initialize_model(model_path):
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        raise FileNotFoundError("Model weights file not found")
    
    model = YOLO(model_path)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

# 检测-追踪
def process_frame(frame, model, tracker, conf, class_id):
    results = model(frame, verbose=False)[0]
    if results.obb:
        # deepsort算法的入参中检测框格式只能是xywh，将obb转换为其最小外接矩形的xyxy坐标
        bboxes = results.obb.xyxy.tolist()
        predict_boxes = results.obb.xyxyxyxy.tolist()
        cls_ls = results.obb.cls.tolist()
        conf_ls = results.obb.conf.tolist()
    else:
        bboxes = results.boxes.xyxy.tolist()
        predict_boxes = results.boxes.xyxy.tolist()
        cls_ls = results.boxes.cls.tolist()
        conf_ls = results.boxes.conf.tolist()

    detections = []
    for i in range(len(bboxes)):
        cls_id, confidence, bbox = int(cls_ls[i]), conf_ls[i], bboxes[i]
        x1, y1, x2, y2 = map(int, bbox)
        
        if class_id is None:
            if confidence < conf:
                continue
        else:
            if cls_id != class_id or confidence < conf:
                continue
        
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, cls_id])
    
    tracks = tracker.update_tracks(detections, frame=frame)
    return predict_boxes, tracks

# 以原坐标画框，因为obb追踪时转换为了其最小外接矩形的xywh坐标
def draw_tracks(frame, predict_boxes, tracks, class_names, colors, class_counters, track_class_mapping, blur_id):
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        class_id = track.get_det_class()

        # 追踪结束后，DeepSORT 输出的追踪结果与输入的水平框具有相同的索引顺序，因此可以直接利用映射关系恢复到旋转框
        if i > len(predict_boxes) - 1:
            continue
        # 画框
        box = predict_boxes[i]
        color = colors[class_id]
        if type(box[0]) == list:
            # todo: 为什么显示逻辑会影响追踪效果？
            # 显示obb的最小外接矩形
            # ltrb = track.to_ltrb()
            # class_id = track.get_det_class()
            # x1, y1, x2, y2 = map(int, ltrb)
            # utils.drwa_bboxes_by_xyxy(frame, [x1, y1, x2, y2], color)

            # 显示obb
            x1, y1 = map(int, box[0])
            x2, y2 = map(int, box[2])
            utils.draw_bboxes_by_obb(frame, box, color)
        else:
            x1, y1, x2, y2 = list(map(int, box))
            utils.drwa_bboxes_by_xyxy(frame, box, color)

        # Assign a new class-specific ID if the track_id is seen for the first time
        if track_id not in track_class_mapping:
            class_counters[class_id] += 1
            track_class_mapping[track_id] = class_counters[class_id]

        class_specific_id = track_class_mapping[track_id]
        text = f"{class_specific_id} - {class_names[class_id]}"
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if blur_id is not None and class_id == blur_id:
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)
    
    return frame


def predict_and_track_video(model_path, video_path, output=None, conf=0.5, bluf_id=None, class_id=None):
    try:
        cap = initialize_video_capture(video_path)
        model = initialize_model(model_path)
        # 先预测一帧拿到类别名列表
        ret, frame = cap.read()
        if ret:
            class_names = model.predict(frame)[0].names
        else:
            class_names = []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

        tracker = DeepSort(max_age=20, n_init=3)

        colors = utils.gene_colors(class_names)

        class_counters = defaultdict(int)
        track_class_mapping = {}
        frame_count = 0

        while True:
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break

            predict_boxes, tracks = process_frame(frame, model, tracker, conf, class_id)
            frame = draw_tracks(frame, predict_boxes, tracks, class_names, colors, class_counters, track_class_mapping, bluf_id)

            end = datetime.datetime.now()
            logger.info(f"Time to process frame {frame_count}: {(end - start).total_seconds():.2f} seconds")
            frame_count += 1

            fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            if output is not None:
                writer.write(frame)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("YOLOv11 Object tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        logger.info("Class counts:")
        for class_id, count in class_counters.items():
            logger.info(f"{class_names[class_id]}: {count}")

    except Exception as e:
        logger.exception("An error occurred during processing")
    finally:
        cap.release()
        if output is not None:
            writer.release()
        cv2.destroyAllWindows()

def main(_argv):
    predict_and_track_video(FLAGS.model_path, FLAGS.video, FLAGS.output, FLAGS.conf, FLAGS.blur_id, FLAGS.class_id)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
