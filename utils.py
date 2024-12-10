import cv2
from ultralytics import YOLO


def show_bboxes(im, obbs, cls_ls, classes, color=(0, 255, 0), thickness=2):
    for i in range(len(obbs)):
        obb = obbs[i]
        cls_id = cls_ls[i]
        # 画旋转框
        x1, y1 = obb[0]
        x2, y2 = obb[1]
        x3, y3 = obb[2]
        x4, y4 = obb[3]
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2),), color, thickness)
        cv2.line(im, (int(x2), int(y2)), (int(x3), int(y3),), color, thickness)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4),), color, thickness)
        cv2.line(im, (int(x4), int(y4)), (int(x1), int(y1),), color, thickness)
        # 显示分类
        cv2.putText(im, classes[cls_id], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)


    # 长、宽等比例缩小到原来的0.5倍
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("output", im)


def predict_img(model_path, img_path):
    model = YOLO(model_path)
    im = cv2.imread(img_path)
    results = model.predict(im)[0]
    # 预测旋转框
    obbs = results.obb.xyxyxyxy.tolist()
    # 分类映射和分类预测list
    names = results.names
    cls_ids = results.obb.cls.tolist()
    show_bboxes(im, obbs, cls_ids, names)
    cv2.waitKey(0)


def predict_video(model_path, video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, im = cap.read()
        if not ret:
            break
        results = model.predict(im)[0]
        obbs = results.obb.xyxyxyxy.tolist()
        names = results.names
        cls_ids = results.obb.cls.tolist()
        show_bboxes(im, obbs, cls_ids, names)
        # 按esc退出
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        else:
            cv2.waitKey(0)
    cap.release()