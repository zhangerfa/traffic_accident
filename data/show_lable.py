import cv2
from detract2yolo import cibver_yolo_txt

path = r'F:\data\UAV-ROD'
data = r'DJI_0006_000000'

im_path = rf'{path}\images\train\{data}.jpg'
xml_path = rf'{path}\Annotations\train\{data}.xml'
im = cv2.imread(im_path)

# 将xml格式转化为yolo_txt格式
res = cibver_yolo_txt(xml_path, ['car'])
# 将标注框画在图片上
for line in res.split('\n'):
    if line:
        cls_id, x1, y1, x2, y2, x3, y3, x4, y4 = [float(a) for a in line.split()]
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2),), (0, 255, 0), 2)
        cv2.line(im, (int(x2), int(y2)), (int(x3), int(y3),), (0, 255, 0), 2)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4),), (0, 255, 0), 2)
        cv2.line(im, (int(x4), int(y4)), (int(x1), int(y1),), (0, 255, 0), 2)
        # break

# 长、宽等比例缩小到原来的0.5倍
im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("output", im)
cv2.waitKey(0)
