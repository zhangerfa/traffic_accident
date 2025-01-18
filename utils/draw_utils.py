"""
展示检测结果的工具函数
"""
import cv2
import numpy as np

def draw_trace_on_frame(trace_dict, frame):
    """
    将车辆轨迹绘制到传入的帧上
    trace_ls: 车辆轨迹: {obj_id: [[x, y, w, h, angle],...],...}
    """
    for obj_id, trace_ls in trace_dict.items():
        for trace in trace_ls:
            box = trace[0]
            x, y, w, h, angle = box
            # 轨迹用一系列点表示，点用圆圈表示
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def draw_box(frame, box, text, color):
    """
    frame: 原图
    box: 检测框，可以是xyxy或者obb
    text: 检测框上显示的文本
    colors: 对象颜色
    """
    if type(box[0]) == list:
        # xyxyxyxy格式的obb
        x1, y1 = map(int, box[0])
        __draw_bboxes_by_obb(frame, box, color)
    else:
        # xyxy格式的bbox
        x1, y1, x2, y2 = list(map(int, box))
        __draw_bboxes_by_xyxy(frame, box, color)

    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def gene_colors(classes):
    """
    为不同的类别随机设置不同的颜色
    """
    if classes is None:
        return []
    np.random.seed(13)
    colors = np.random.randint(0, 255, size=(len(classes), 3))
    return [list(map(int, color)) for color in colors]

def show_labels(img_path, xml_path):
    from utils.data_utils import cibver_yolo_txt

    im = cv2.imread(img_path)

    # 将xml格式转化为yolo_txt格式
    class_names = ['car']
    res = cibver_yolo_txt(xml_path, class_names, False)
    # 将标注框画在图片上
    for line in res.split('\n'):
        if line:
            cls_id, x1, y1, x2, y2, x3, y3, x4, y4 = [float(a) for a in line.split()]
            draw_box(im, (x1, y1, x2, y2, x3, y3, x4, y4), class_names[int(cls_id)], (0, 255, 0))

    # 长、宽等比例缩小到原来的0.5倍
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("output", im)
    cv2.waitKey(0)


def __draw_bboxes_by_xyxy(im, xyxy, color, thickness=2):
    """
    画xyxy格式的检测框
    """
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def __draw_bboxes_by_obb(im, obb, color, thickness=2):
    """
    画xyxyxyxy格式的检测框
    """
    # 画旋转框
    x1, y1 = obb[0]
    x2, y2 = obb[1]
    x3, y3 = obb[2]
    x4, y4 = obb[3]
    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2),), color, thickness)
    cv2.line(im, (int(x2), int(y2)), (int(x3), int(y3),), color, thickness)
    cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4),), color, thickness)
    cv2.line(im, (int(x4), int(y4)), (int(x1), int(y1),), color, thickness)