from roboflow import Roboflow
rf = Roboflow(api_key="SuklAeRTbtD1k4yDlJsC")
project = rf.workspace().project("bird-nest-exr6l")
model = project.version(2).model

# infer on a local image
# print(model.predict("Test.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("Test.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())