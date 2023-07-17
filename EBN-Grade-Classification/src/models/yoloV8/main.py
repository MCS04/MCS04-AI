from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml") # build a new model from scratch

## Don't use this first
# Use the model
results = model.train(data="config.yaml", epochs=1)  # train the model