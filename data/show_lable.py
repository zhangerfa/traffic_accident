import cv2

path = r'F:\data\UAV-ROD'
data = r'DJI_0006_000000'

im_path = rf'{path}\images\train\{data}.jpg'
label_path = rf'{path}\labels\train\{data}.txt'
im = cv2.imread(im_path)

# 读取标签文件
with open(label_path, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        label = line.strip().split(' ')
        cla, cx, cy, w, h, angle = map(float, label)
        # 将 cx、cy、w、h、angle描述的旋转框画在图片im中
        box = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(box)
        box = box.astype(int)
        cv2.polylines(im, [box], True, (0, 255, 0), 2)


# 长、宽等比例缩小到原来的0.5倍
im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("output", im)
cv2.waitKey(0)
