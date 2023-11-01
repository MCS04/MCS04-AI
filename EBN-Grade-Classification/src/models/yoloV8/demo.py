from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2

# Import the model
model = YOLO("yolov8n.yaml")
# results = model.predict(source="0", show=True)
# print (results)

# Train the model
results = model.train(data="data.yaml",epochs=2)