import os

from ultralytics import YOLO

if __name__ == '__main__':
    # 设置环境变量来避免重复的 libiomp5md.dll 错误
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    model = YOLO("yolo11n-obb.yaml")

    # epoch是数据集的迭代次数，也就是数据集被学习几次，数值越小训练速度越快，数值越大对数据学习程度越好，但可能过拟合
    # batch是每次迭代的数据量，也就是一次加载的数据个数，数值越大训练速度越快，数值越小训练速度越慢，受限于显存大小
    # imgsz是图片的大小，如果输入图片大小不一致，yolo内部会自动调整大小，imgsz越大训练速度越慢，但数值小可能影响精度，因为图片压缩会导致信息丢失
    # workers是数据加载的线程数，受限于cpu核心数，数值越大训练速度越快，数值越小训练速度越慢
    model.train(data=r'data/type.yaml', epochs=300, batch=64, imgsz=416, workers=4)