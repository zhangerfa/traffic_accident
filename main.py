from incident.incident_processor import IncidentProcessor
from incident.video_processor import VideoProcessor

if __name__ == "__main__":
    yolo_weight_path = 'weights/yolo11x-obb.pt'
    video_path = r"D:\download\DJI_0802.MP4"
    # video_path = r"D:\programs\OneDrive\文档\毕设\逆行.mp4"
    img_path = r"E:\data\UAV-ROD\images\train\DJI_0006_000000.jpg"

    trace_path = r"D:\download\trace.csv"

    video_processor = VideoProcessor(yolo_weight_path, video_path, 0)
    lane_space = IncidentProcessor(video_processor)

    # 识别、追踪、轨迹展示
    # video_processor.predict_and_track()
    # video_processor.predict()
    # video_processor.show_trace()

    # 提取车辆轨迹并保存为csv文件
    # lane_space.to_csv(trace_path, 200)

    # 从csv文件中读取轨迹数据
    # lane_space.load_trace_from_csv(trace_path)

    # 将前200帧的轨迹数据画到第0帧图片上
    # video_processor.show_trace_on_frame(0, 200)

    # 从轨迹中计算第100帧中的车道行驶方向
    # lane_direction = lane_space.cal_lane_direction(100)
    # print(f"车道行驶方向：{lane_direction}")
    # video_processor.show_speed_cluster(100)

    # 获取前100帧的交通事件
    traffic_incidents = lane_space.get_traffic_incidents_offline(100)
    print(f"交通事件：{traffic_incidents}")
