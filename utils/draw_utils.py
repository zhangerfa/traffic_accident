"""
展示检测结果的工具函数
"""

import cv2
import numpy as np

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
