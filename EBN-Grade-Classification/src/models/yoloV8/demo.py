from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2

model = YOLO("yolov8n.yaml")
# results = model.predict(source="0", show=True)
# print (results)
results = model.train(data="data.yaml",epochs=2)