from utils.draw_utils import draw_trace_on_frame, draw_speed_cluster
from utils.trace_utils import extract_trace_to_csv, load_trace_from_csv, cal_lane_from_trace, \
    extract_speed_vector_from_trace, cluster_speed_vector
from video_processor import VideoProcessor

if __name__ == "__main__":
    yolo_weight_path = 'weights/yolo11x-obb.pt'
    # video_path = r"E:\data\self\trace\DJI_0802.MP4"
    # video_path = r"D:\download\test.mp4"
    video_path = r"D:\download\DJI_0802.MP4"
    # video_path = r"D:\programs\OneDrive\文档\毕设\逆行.mp4"
    img_path = r"E:\data\UAV-ROD\images\train\DJI_0006_000000.jpg"

    trace_path = r"D:\download\trace.csv"

    video_processor = VideoProcessor(yolo_weight_path, video_path, 0.4)

    # 识别、追踪、轨迹展示
    # video_processor.predict_and_track()
    # video_processor.predict()
    # video_processor.show_trace()

    # 提取车辆轨迹并保存为csv文件
    # trace_dict = video_processor.extract_trace(200)
    # extract_trace_to_csv(trace_dict, trace_path)

    # 从scv文件中读取轨迹数据
    trace_dict = load_trace_from_csv(trace_path)

    # 从csv中读取轨迹数据画到第0帧图片上
    # first_frame = video_processor.get_frame(0)
    # draw_trace_on_frame(trace, first_frame)
    # cv2.imshow('first_frame', first_frame)
    # cv2.waitKey(0)

    # 从轨迹中计算车道行驶方向
    # lane_direction = cal_lane_from_trace(trace_dict, 100)
    # print(f"车道行驶方向：{lane_direction}")
    # 查看车辆速度矢量的聚类情况
    speed_ls = extract_speed_vector_from_trace(trace_dict, 100)
    cluster_dict = cluster_speed_vector(speed_ls)
    draw_speed_cluster(cluster_dict)