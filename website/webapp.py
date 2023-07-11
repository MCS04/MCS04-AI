from enum import Enum
import gradio as gr
import random as rand
import keras
import asyncio
import numpy as np

class Model(Enum):
    RESNET50 = "ResNet50"
    YOLOV8 = "Yolov8"

# Put ML Model here


def resnet50(img):
    img = np.expand_dims(img, axis=0)
    model = keras.models.load_model(
        "../EBN-Grade-Classification/src/models/resnet50/resnet50model")
    prediction = model.predict(img)
    labels = ["A", "B", "C"]
    max_index = np.argmax(prediction)
    class_label = labels[max_index]
    return class_label


def classify(img, model):
    result = ""
    if model == Model.RESNET50.value:
        result = resnet50(img)

    elif model == Model.YOLOV8:
        result = ""

    return f"This is Grade {result}" 

radio = gr.Radio([Model.RESNET50, Model.YOLOV8], label="Models")
inputs = [gr.Image(
    shape=(256, 256)), radio]
demo = gr.Interface(fn=classify, inputs=inputs,
                    outputs=gr.Label(num_top_classes=3))


if __name__ == "__main__":
    demo.launch()
