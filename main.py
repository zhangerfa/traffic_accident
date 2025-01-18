import cv2

from utils.draw_utils import draw_trace_on_frame
from utils.trace_utils import extract_trace_to_csv, load_trace_from_csv
from video_processor import VideoProcessor

if __name__ == "__main__":
    yolo_weight_path = 'weights/yolo11x-obb.pt'
    video_path = r"D:\download\test.mp4"

    img_path = r"E:\data\UAV-ROD\images\train\DJI_0006_000000.jpg"

    trace_path = r"D:\download\trace.csv"

    video_processor = VideoProcessor(yolo_weight_path, video_path, 0.6)

    # 提取轨迹并保存到csv
    # extract_trace_to_csv(video_processor.obb_tracker.trace_dict, trace_path)

    # 识别、追踪、轨迹展示
    # video_processor.predict_and_track()
    # video_processor.predict()
    video_processor.show_trace()

    # 从csv中读取轨迹数据画到第0帧图片上
    # first_frame = video_processor.get_frame(0)
    # draw_trace_on_frame(load_trace_from_csv(trace_path), first_frame)
    # cv2.imshow('first_frame', first_frame)
    # cv2.waitKey(0)
