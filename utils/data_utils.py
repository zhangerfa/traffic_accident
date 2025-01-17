"""
数据处理和展示工具函数
"""

# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import glob

import cv2
import numpy as np

from utils.draw_utils import draw_box


def show_labels(img_path, xml_path):
    im = cv2.imread(img_path)

    # 将xml格式转化为yolo_txt格式
    class_names = ['car']
    res = __cibver_yolo_txt(xml_path, class_names, False)
    # 将标注框画在图片上
    for line in res.split('\n'):
        if line:
            cls_id, x1, y1, x2, y2, x3, y3, x4, y4 = [float(a) for a in line.split()]
            draw_box(im, (x1, y1, x2, y2, x3, y3, x4, y4), class_names[int(cls_id)], (0, 255, 0))

    # 长、宽等比例缩小到原来的0.5倍
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("output", im)
    cv2.waitKey(0)


def detract2yolo(path, classes):
    for set_name in ["test", "train"]:
        annotation_path = path + r"\Annotations"
        images_path = path + r"\images"
        labels_path = path + r"\labels"
        # 获取数据集所有图片名的txt文件
        # __get_image_txt(images_path + fr"\{set_name}", path, set_name)
        # 将annotation文件中信息转换为yolo_txt格式
        # 分类
        __convert_annotation(annotation_path + fr"\{set_name}", labels_path + rf"\{set_name}",
                             path + rf"\{set_name}.txt", classes)

def xyxyxyxy_to_xywha(box):
    """
    将矩形的四个角点坐标转换为中心 (cx, cy)、宽度 w、高度 h 和角度 angle
    输入坐标的四个角点顺序为：左上角、右上角、右下角、左下角
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box

    len1 = __get_length(x1, y1, x2, y2)
    len2 = __get_length(x2, y2, x3, y3)

    # 对角线的中点是矩形的中心
    cx = (x1 + x3) / 2
    cy = (y1 + y3) / 2

    # 角度长边与x轴正方向（向右）的夹角，长边方向是从左上角到右下角或从右上角到左下角
    if len1 > len2:
        angle = np.arctan2(y1 - y2, x1 - x2)
        w = len1
        h = len2
    else:
        angle = np.arctan2(y2 - y3, x2 - x3)
        w = len2
        h = len1

    return cx, cy, w, h, angle

def xywha_to_xyxyxyxy(cx, cy, w, h, angle, image_width, image_height):
    """
    将矩形的中心 (cx, cy)、宽度 w、高度 h 和角度 angle 转换为四个角点坐标。
    角度 angle 单位为弧度
    """

    # 计算矩形四个角点的相对位置（相对于中心点）
    half_w = w / 2
    half_h = h / 2

    # 定义矩形的四个角点（相对于中心）
    corners = [
        (cx - half_w, cy - half_h),  # 左上角
        (cx + half_w, cy - half_h),  # 右上角
        (cx + half_w, cy + half_h),  # 右下角
        (cx - half_w, cy + half_h)  # 左下角
    ]

    # 对四个角点进行旋转
    rotated_corners = []
    for corner in corners:
        x, y = corner
        rotated_x, rotated_y = __rotate_point(x, y, cx, cy, angle, image_width, image_height)
        rotated_corners.append((rotated_x, rotated_y))

    # 返回四个角点的坐标（x1, y1, x2, y2, x3, y3, x4, y4）
    return rotated_corners[0][0], rotated_corners[0][1], \
        rotated_corners[1][0], rotated_corners[1][1], \
        rotated_corners[2][0], rotated_corners[2][1], \
        rotated_corners[3][0], rotated_corners[3][1]

def __get_length(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def __rotate_point(x, y, cx, cy, angle_rad, img_width, img_height):
    """
    旋转点 (x, y) 关于中心点 (cx, cy) 旋转角度 angle_rad，
    确保坐标不超过图像边界 (img_width, img_height)
    """
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # 旋转矩阵应用
    x_rot = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
    y_rot = sin_angle * (x - cx) + cos_angle * (y - cy) + cy

    # 确保坐标在图像范围内
    x_rot = max(0, min(x_rot, img_width - 1))
    y_rot = max(0, min(y_rot, img_height - 1))

    return x_rot, y_rot

# 坐标归一化：yolo中的坐标是相对于图像宽度和高度的，范围是[0, 1]
def __normalize_coordinates(bbox, image_width, image_height):
    # 归一化坐标，除以图像宽度和高度
    normalized_bbox = []
    for i in range(8):
        normalized_bbox.append(bbox[i] / image_width if i % 2 == 0 else bbox[i] / image_height)
    return normalized_bbox

def __cibver_yolo_txt(xml_path, classes, normalize=True):
    # 读入xml文件
    in_file = open(xml_path, encoding='UTF-8')
    # 信息提取
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    res = ""
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('robndbox')
        cx = float(xmlbox.find('cx').text)
        cy = float(xmlbox.find('cy').text)
        cw = float(xmlbox.find('w').text)
        ch = float(xmlbox.find('h').text)
        angle = float(xmlbox.find('angle').text)
        # 将cx, cy, w, h, angle转换为x1, y1, x2, y2, x3, y3, x4, y4
        bbox = xywha_to_xyxyxyxy(cx, cy, cw, ch, angle, image_width, image_height)
        # 归一化坐标
        if normalize:
            bbox = __normalize_coordinates(bbox, image_width, image_height)
        # 一行转换为yolo_txt格式
        res += str(cls_id) + " " + " ".join([str(a) for a in bbox]) + '\n'
    return res

# 读入xml文件提取标准信息并保存为yolo_txt格式
# in_path: Annotation文件所在路径
# out_path: yolo_txt文件存储位置，一般命名为 labels
# set_txt_path: 存储数据集中数据名（不包含后缀）的txt文件的路径
# classes: 类别list
def __convert_annotation(in_path, out_path, set_txt_path, classes):
    # out_path若不存在则创建
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 遍历所有xml文件，提取信息存储为yolo_txt文件
    img_ids = open(set_txt_path).read().strip().split()
    for img_id in img_ids:
        # 创建yolo_txt文件
        out_file = open(out_path + f'/{img_id}.txt', 'w')
        # 转换为yolo_txt格式
        res = __cibver_yolo_txt(in_path + f'/{img_id}.xml', classes)
        out_file.write(res)


# 获取数据集所有图片名的txt文件
# in_path: 图片所在路径
# out_path: 存放txt文件的路径
# set_name: 数据集名称，txt文件将命名为：set_name.txt
def __get_image_txt(in_path, out_path, set_name):
    # 获取文件夹中所有图片文件的路径
    file_paths = glob.glob(os.path.join(in_path, '*.jpg'))

    # 打开要写入的 txt 文件
    with open(out_path + rf'\{set_name}.txt', 'w') as f:
        # 遍历文件路径列表，写入文件名到 txt 文件中
        for file_path in file_paths:
            # 获取文件名（不包含后缀）
            filename = os.path.splitext(os.path.basename(file_path))[0]
            # 写入文件名到 txt 文件中
            f.write(filename + '\n')
    f.close()