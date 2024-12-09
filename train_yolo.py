from ultralytics import YOLO

model = YOLO("yolo11n-obb.yaml")

model.train(data=r'data/type.yaml', epochs=500, batch=256, imgsz=640)