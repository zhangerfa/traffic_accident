"""
yolo预测工具函数
"""
import cv2

from incident.obb_tracker import parse_yolo_result, initialize_yolo
from incident.utils.draw_utils import gene_colors, draw_box


def predict_img(yolo_weight_path, img_path):
    """
    预测并展示图片，按q键退出
    """
    yolo = initialize_yolo(yolo_weight_path)
    frame = cv2.imread(img_path)
    colors = gene_colors(yolo.names)
    predict_and_show_frame(yolo, frame, colors)

    # 按q键退出
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

    
def predict_and_show_frame(yolo, frame, colors):
    results = yolo.predict(frame)[0]
    predict_boxes, _, cls_ls, _ = parse_yolo_result(results)
    for obb, cls_id in zip(predict_boxes, cls_ls):
        draw_box(frame, obb, yolo.names[cls_id], colors[int(cls_id)])

    # 长、宽等比例缩小到原来的0.5倍
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("output", frame)

