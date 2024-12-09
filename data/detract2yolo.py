# -*- coding: utf-8 -*-
import math
import xml.etree.ElementTree as ET
import os
import glob

# 坐标归一化：yolo中的坐标是相对于图像宽度和高度的，范围是[0, 1]
def normalize_coordinates(x, y, w, h, image_width, image_height):
    # 将坐标和尺寸转换为浮点型
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    # 归一化坐标，除以图像宽度和高度
    x_normalized = max(x / image_width, 0)
    y_normalized = max(y / image_height, 0)
    w_normalized = max(w / image_width, 0)
    h_normalized = max(h / image_height, 0)
    # 返回归一化后的坐标元组
    return x_normalized, y_normalized, w_normalized, h_normalized

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
        # 读入xml文件
        in_file = open(in_path + f'/{img_id}.xml', encoding='UTF-8')
        # 创建yolo_txt文件
        out_file = open(out_path + f'/{img_id}.txt', 'w')
        # 信息提取
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
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
            # 弧度转换为角度
            angle = float(xmlbox.find('angle').text) * 180 / math.pi
            # 标注越界修正
            if cw > w:
                cw = w
            if ch > h:
                ch = h
            if cx > w:
                cx = w
            if cy > h:
                cy = h
            # 归一化坐标
            cx, cy, cw, ch = normalize_coordinates(cx, cy, cw, ch, w, h)
            bb = [cx, cy, cw, ch, angle]
            # 写入yolo_txt文件
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


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