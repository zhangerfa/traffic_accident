import cv2
import numpy as np
from ultralytics import YOLO

colors = []

# 为不同的类别随机设置不同的颜色
def gene_colors(classes):
    np.random.seed(13)
    colors = np.random.randint(0, 255, size=(len(classes), 3))
    return [list(map(int, color)) for color in colors]


def drwa_bboxes_by_xyxy(im, xyxy, color, thickness=2):
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_bboxes_by_obb(im, obb, color, thickness=2):
    # 画旋转框
    x1, y1 = obb[0]
    x2, y2 = obb[1]
    x3, y3 = obb[2]
    x4, y4 = obb[3]
    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2),), color, thickness)
    cv2.line(im, (int(x2), int(y2)), (int(x3), int(y3),), color, thickness)
    cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4),), color, thickness)
    cv2.line(im, (int(x4), int(y4)), (int(x1), int(y1),), color, thickness)

# 根据obb或者xyxy的结果画出结果
def __draw_results(im, results, classes):
    names = results.names
    # 判断是否是obb
    if results.obb:
        is_obb = True
    else:
        is_obb = False
    if is_obb:
        obbs = results.obb.xyxyxyxy.tolist()
        cls_ids = results.obb.cls.tolist()
        for obb, cls_id in zip(obbs, cls_ids):
            color = colors[int(cls_id)]
            draw_bboxes_by_obb(im, obb, color)
            # 显示分类
            cv2.putText(im, classes[cls_id], (int(obb[0][0]), int(obb[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        boxes = results.boxes
        xyxys = boxes.xyxy.tolist()
        cls_ids = boxes.cls.tolist()
        for xyxy, cls_id in zip(xyxys, cls_ids):
            color = colors[int(cls_id)]
            drwa_bboxes_by_xyxy(im, xyxy, color)
            cv2.putText(im, classes[cls_id], (int(xyxy[0]), int(xyxy[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def predict_img(model_path, img_path):
    model = YOLO(model_path)
    im = cv2.imread(img_path)
    results = model.predict(im)[0]
    # 分类映射和分类预测list
    names = results.names
    global colors
    colors = gene_colors(names)
    __draw_results(im, results, names)

    # 长、宽等比例缩小到原来的0.5倍
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("output", im)
    cv2.waitKey(0)


def predict_video(model_path, video_path, is_obb=True):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    ret, im = cap.read()
    # 预测一帧拿到类别名列表, 来为每个类型设置颜色
    names = model.names
    if ret:
        global colors
        colors = gene_colors(names)
    while True:
        ret, im = cap.read()
        if not ret:
            break
        results = model.predict(im)[0]
        __draw_results(im, results, names)

        # 长、宽等比例缩小到原来的0.5倍
        im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("output", im)
        cv2.waitKey(0)
    cap.release()