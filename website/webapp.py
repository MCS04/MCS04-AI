from enum import Enum
import gradio as gr
import random as rand
import tensorflow as tf
from tensorflow import keras
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

def yolov8(vid):
    return vid


def classify(img, model):
    result = ""
    if model == Model.RESNET50.value:
        result = resnet50(img)

    elif model == Model.YOLOV8:
        result = ""

    return f"This is Grade {result}" 


with gr.Blocks() as demo:
    gr.Markdown("Choose your model")
    with gr.Tab("ResNet50"):
        with gr.Row():
            resnet_input = gr.Image(shape=(256, 256))
            resnet_output = gr.Label(num_top_classes=3)    # missing path file /resnet50model
        resnet_submit_button = gr.Button("Submit")
    with gr.Tab("Yolov8"):
        with gr.Row():
            yolov8_input = gr.Video(source="webcam")    # add streaming=True or live=True? 
            yolov8_output = gr.Label()
        yolov8_button = gr.Button("Submit")

    # resnet_submit_button.click(resnet50, inputs=resnet_input, outputs=resnet_output)
    # yolov8_button.click(yolov8, inputs=yolov8_input, outputs=yolov8_output)

if __name__ == "__main__":
    demo.launch()


# if want to test just uncomment and use this
"""
radio = gr.Radio([Model.RESNET50, Model.YOLOV8], label="Models")
inputs = [gr.Image(
    shape=(256, 256)), radio]
demo = gr.Interface(fn=classify, inputs=inputs,
                    outputs=gr.Label(num_top_classes=3))
"""