# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import glob

import numpy as np


def rotate_point(x, y, cx, cy, angle_rad):
    """
    旋转点 (x, y) 关于中心点 (cx, cy) 旋转角度 angle_rad
    """
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # 旋转矩阵应用
    x_rot = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
    y_rot = sin_angle * (x - cx) + cos_angle * (y - cy) + cy

    return x_rot, y_rot


def convert_to_bbox(cx, cy, w, h, angle):
    """
    将矩形的中心 (cx, cy)、宽度 w、高度 h 和角度 angle 转换为四个角点坐标。
    角度 angle 单位为弧度
    """

    # 计算矩形四个角点的相对位置（相对于中心点）
    half_w = w / 2
    half_h = h / 2

    # 定义矩形的四个角点（相对于中心）
    corners = [
        (-half_w, -half_h),  # 左上角
        (half_w, -half_h),  # 右上角
        (half_w, half_h),  # 右下角
        (-half_w, half_h)  # 左下角
    ]

    # 对四个角点进行旋转
    rotated_corners = []
    for corner in corners:
        x, y = corner
        rotated_x, rotated_y = rotate_point(x, y, cx, cy, angle)
        rotated_corners.append((rotated_x, rotated_y))

    # 返回四个角点的坐标（x1, y1, x2, y2, x3, y3, x4, y4）
    return rotated_corners[0][0], rotated_corners[0][1], \
        rotated_corners[1][0], rotated_corners[1][1], \
        rotated_corners[2][0], rotated_corners[2][1], \
        rotated_corners[3][0], rotated_corners[3][1]

# 坐标归一化：yolo中的坐标是相对于图像宽度和高度的，范围是[0, 1]
def normalize_coordinates(bbox, image_width, image_height):
    # 归一化坐标，除以图像宽度和高度
    normalized_bbox = []
    for i in range(8):
        normalized_bbox.append(bbox[i] / image_width if i % 2 == 0 else bbox[i] / image_height)
    return normalized_bbox

def cibver_yolo_txt(xml_path, classes):
    # 读入xml文件
    in_file = open(xml_path, encoding='UTF-8')
    # 信息提取
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

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
        # 标注越界修正
        if cw > w:
            cw = w
        if ch > h:
            ch = h
        if cx > w:
            cx = w
        if cy > h:
            cy = h
        # 将cx, cy, w, h, angle转换为x1, y1, x2, y2, x3, y3, x4, y4
        bbox = convert_to_bbox(cx, cy, cw, ch, angle)
        # 归一化坐标
        # bbox = normalize_coordinates(bbox, w, h)
        # 一行转换为yolo_txt格式
        res += str(cls_id) + " " + " ".join([str(a) for a in bbox]) + '\n'
    return res

# 读入xml文件提取标准信息并保存为yolo_txt格式
# in_path: Annotation文件所在路径
# out_path: yolo_txt文件存储位置，一般命名为 labels
# set_txt_path: 存储数据集中数据名（不包含后缀）的txt文件的路径
# classes: 类别list
def convert_annotation(in_path, out_path, set_txt_path, classes):
    # out_path若不存在则创建
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 遍历所有xml文件，提取信息存储为yolo_txt文件
    img_ids = open(set_txt_path).read().strip().split()
    for img_id in img_ids:
        # 创建yolo_txt文件
        out_file = open(out_path + f'/{img_id}.txt', 'w')
        # 转换为yolo_txt格式
        res = cibver_yolo_txt(in_path + f'/{img_id}.xml', classes)
        out_file.write(res)


# 获取数据集所有图片名的txt文件
# in_path: 图片所在路径
# out_path: 存放txt文件的路径
# set_name: 数据集名称，txt文件将命名为：set_name.txt
def get_image_txt(in_path, out_path, set_name):
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


if __name__ == "__main__":
    for set_name in ["test", "train"]:
        path = rf"F:\data\UAV-ROD"
        annotation_path = path + r"\Annotations"
        images_path = path + r"\images"
        labels_path = path + r"\labels"
        # 获取数据集所有图片名的txt文件
        # get_image_txt(images_path + fr"\{set_name}", path, set_name)
        # 将annotation文件中信息转换为yolo_txt格式
        # 分类
        classes = ["car"]
        convert_annotation(annotation_path + fr"\{set_name}",
                           labels_path + rf"\{set_name}",
                           path + rf"\{set_name}.txt", classes)