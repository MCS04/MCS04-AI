# import sys

# sys.path.insert(1, 'C:\Users\SC Wintech\Documents\mcs04\MCS04-AI\EBN-Grade-Classification')

# Standard imports
import gradio as gr
from enum import Enum
import numpy as np
import random as rand
import asyncio


# Import for deep learning purposes
import tensorflow as tf
from tensorflow import keras
from roboflow import Roboflow

# test
import requests

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Gradio library import


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


def yolov8(img_path):
    print("Running yolov8!")
    rf = Roboflow(api_key="SuklAeRTbtD1k4yDlJsC")
    project = rf.workspace().project("bird-nest-exr6l")
    model = project.version(2).model

    # infer on a local image
    # print(model.predict("Test.jpg", confidence=40, overlap=30).json())

    # visualize your prediction
    # model.predict("Test.jpg", confidence=40, overlap=30).save("prediction.jpg")

    predictions = model.predict(img_path, confidence=40, overlap=30)
    predictions_json = predictions.json

    print(predictions_json)

    for bounding_box in predictions:
        # x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
        # x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
        # y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
        # y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
        class_name = bounding_box['class']
        confidence_score = bounding_box['confidence']

        detection_results = bounding_box
        print(detection_results)

        grade_name = switch_grade(class_name)
        class_and_confidence = {grade_name: confidence_score}
        print(class_and_confidence, '\n')

    predictions.save("prediction.jpg")

    return class_and_confidence

    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


def switch_grade(key):
    if key == "0":
        return "Grade A"
    elif key == "1":
        return "Grade B"
    elif key == "2":
        return "Grade C"


# test mock model
inception_net = tf.keras.applications.MobileNetV2()


def test_model(inp):
    print("Running mobile net!")
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return [confidences, confidences]


def classify(img, model):
    print("Classified model!")
    # print(img)
    # print(model)
    result = ""
    if model == Model.RESNET50.value:
        # print("Running mobilenet")
        result = test_model(img)

    elif model == Model.YOLOV8.value:
        # print("Running yolov8")
        result = [yolov8(img), "prediction.jpg", ]

    else:
        print("no value match found")

    return result


def test(a):
    try:
        # Validate input (e.g., check if input_image is not None and of the correct type)
        if a is None:
            return "Invalid input: Image is missing or empty."
        elif not isinstance(a, str):
            return "Not a string"

        # Replace this with your model's inference code
        return "Success: " + a
    except Exception as e:
        return f"An error occurred: {str(e)}"


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
                image_input = gr.Image(
                    type="filepath", label="Upload EBN Image")
                # , shape=(224, 224)
                # text_input = gr.Textbox()    # for testing purposes
                image_model_input = gr.Dropdown(
                    models, label='Select your Prediction Model')
                image_button = gr.Button("Predict")
            with gr.Column():
                # text_output = gr.Textbox()
                image_output = gr.Label(
                    num_top_classes=3, label="Grade Prediction Output")
                box_output = gr.Image(shape=(56, 56), label="Detected Area")

    with gr.Tab("Real-Time Video Streaming"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Image(source="webcam", streaming=True)
                video_model_input = gr.Dropdown(
                    models, label='Select your Prediction Model')
                video_button = gr.Button("Start Prediction")
            with gr.Column():
                video_output = gr.Label(num_top_classes=3)

    # image_button.click(test, inputs = text_input, outputs = image_output)    # for testing purposes
    image_button.click(classify, inputs=[
                       image_input, image_model_input], outputs=[image_output, box_output])
    # video_button.click(test, inputs=[video_input, video_model_input], outputs=[video_output])

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
