# Standard imports
from enum import Enum
import numpy as np
import random as rand
import asyncio

# Import for deep learning purposes
import tensorflow as tf
from tensorflow import keras

#test
import requests

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Gradio library import
import gradio as gr


class Model(Enum):
    RESNET50 = "resnet50"
    YOLOV8 = "yolov8"

# Put ML Model here
def resnet50(img):
    img = np.expand_dims(img, axis=0)
    model = keras.models.load_model(
        "../EBN-Grade-Classification/src/models/resnet50/resnet50model")
    prediction = model.predict(img)
    labels = ["A", "B", "C"]
    max_index = np.argmax(prediction)
    class_label = labels[max_index]

    return f"This is Grade {class_label}"

def yolov8(vid):
    return vid


# test mock model
inception_net = tf.keras.applications.MobileNetV2()

def test_model(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences



def classify(img, model):
    result = ""
    if model == Model.RESNET50.value:
        result = test_model(img)

    elif model == Model.YOLOV8:
        result = ""

    return result 

def test(a, b):
    return "Model is " + b + "and output is " + a

# image_input = gr.Image(shape=(256, 256))
# video_input = gr.Video(source="webcam")
# model_input_1 = gr.Dropdown(models, value=models[0], label="Model")
# model_input_2 = gr.Dropdown(models, value=models[0], label="Model")

# image_outputs = gr.Label(num_top_classes=3)
# video_outputs = gr.Label(num_top_classes=3)
# title = "Edible Bird's Nest (EBN) Grade Classification System"
# description = "<div align='center'>Created with object detection using deep learning models: YOLOv8 and ResNet50</div>\
#                <div align='center'>Author: Caleb Tan, Christine Chiong, Wong Yi Zhen Nicholas & Brian Nge.</div>"
# article = "<div align='center>Special thanks to Dr Lim Mei Kuan, Dr Chong Chun Yong & Dr Lai Weng Kin</div>"

# upload_image_demo = gr.Interface(
#     fn=test,
#     inputs=[image_input, model_input_1],
#     outputs=image_outputs,
# )

# video_stream_demo = gr.Interface(
#     fn=test2,
#     inputs=[video_input, model_input_2],
#     outputs=video_outputs
# )

# demo = gr.TabbedInterface([upload_image_demo, video_stream_demo], ["Upload Image", "Real-time Video Streaming"], title=title)


models = ['resnet50', 'yolov8']

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <center><h1>Edible Bird's Nest (EBN) Grade Classification System</h1></center>
        <center><p>Created by Caleb Tan, Christine Chiong, Wong Yi Zhen Nicholas & Brian Nge</p></center>
        """
        )

    with gr.Tab("Upload Image"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(shape=(224, 224))
                image_model_input = gr.Dropdown(models, label='Select your Prediction Model')
                image_button = gr.Button("Predict")
            with gr.Column():
                image_output = gr.Textbox()
                # image_output = gr.Label(num_top_classes=3)

    with gr.Tab("Real-Time Video Streaming"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Image(source="webcam", streaming=True)
                video_model_input = gr.Dropdown(models, label='Select your Prediction Model')
                video_button = gr.Button("Start Prediction")
            with gr.Column():
                video_output = gr.Label(num_top_classes=3)
    

    image_button.click(classify, inputs=[image_input, image_model_input], outputs=image_output)
    video_button.click(test, inputs=[video_input, video_model_input], outputs=[video_output])

    gr.Markdown(
        """
        <center><p>Special thanks to Dr Lim Mei Kuan, Dr Chong Chun Yong & Dr Lai Weng Kin</p></center>
        """
    )


if __name__ == "__main__":
    demo.launch(debug=True)


# if want to test just uncomment and use this
"""
radio = gr.Radio([Model.RESNET50, Model.YOLOV8], label="Models")
inputs = [gr.Image(
    shape=(256, 256)), radio]
demo = gr.Interface(fn=classify, inputs=inputs,
                    outputs=gr.Label(num_top_classes=3))
"""

"""
with gr.Blocks() as demo:
    gr.Markdown("Choose your model")
    with gr.Tab("ResNet50"):
        with gr.Row():
            resnet50_input = gr.Image(shape=(256, 256))
            resnet50_output = gr.Label(num_top_classes=3)    
        resnet50_submit_button = gr.Button("Predict")
    with gr.Tab("Yolov8"):
        with gr.Row():
            yolov8_input = gr.Video(source="webcam")    # add streaming=True or live=True? check thru
            yolov8output = gr.Label(num_top_classes=3)                   # with yolo system? fetch grade from rectangle? 
        yolov8_submit_button = gr.Button("Predict")

    resnet50_submit_button.click(resnet50, inputs=resnet50_input, outputs=resnet50_output)
    # yolov8_button.click(yolov8, inputs=yolov8_input, outputs=yolov8_output)
"""