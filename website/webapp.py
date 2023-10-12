# Standard imports
import gradio as gr
from enum import Enum
import numpy as np
import random as rand
import cv2
from PIL import Image


# Import for deep learning purposes
import tensorflow as tf
from tensorflow import keras
from roboflow import Roboflow
from ultralytics import YOLO

# detectron2 stuff
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import cv2
import torch
import torchvision

# test
import requests

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Importing roboflow project
# rf = Roboflow(api_key="SuklAeRTbtD1k4yDlJsC")
# project = rf.workspace().project("bird-nest-exr6l")
# model = project.version(2).model

# Importing deep learning models
model = YOLO('yolov8n.pt')
model = YOLO('weights/best.pt')
model.fuse()

# importing the detectron2 models
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "./weights"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # your number of classes + 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)


class Model(Enum):
    RESNET50 = "resnet50"
    YOLOV8 = "yolov8"
    MOBILENET = "mobilenetv2"
    DETECTRON = "detectron2"


def resnet50(img):

    img = np.expand_dims(img, axis=0)
    model = keras.models.load_model(
        "../EBN-Grade-Classification/src/models/resnet50/resnet50model")
    prediction = model.predict(img)
    labels = ["A", "B", "C"]
    max_index = np.argmax(prediction)
    class_label = labels[max_index]

    return f"This is Grade {class_label}"


def detectron2(img, isVideo):
    if not isVideo:
        img = cv2.imread(img)

        # testing purposes
        # cv2.imshow("widnow", img)
        # key = cv2.waitKey(10000)
        # if key == 27:
        #     cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

    else:
        img = img[:, :, ::-1] # rgb terbalik for video streaming, no need to cv2.imread as it comes in a numpy array

        # testing purposes
        # cv2.imshow("window", img)
        # key = cv2.waitKey(10000)
        # if key == 27:
        #     cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

    outputs = predictor(img)
    print(outputs)

    v = Visualizer(img[:, :, ::-1],
                   scale=0.8
                   )
    class_and_confidence = {}

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # getting the predicted classes
    pred_classes = outputs["instances"].pred_classes
    pred_classes_list = pred_classes.cpu().numpy().tolist()

    # getting the predicted scores
    scores = outputs["instances"].scores
    scores_list = scores.cpu().numpy().tolist()

    # grades dictionary(different from yolov8)
    grade_dict = {1: "Grade A", 2: "Grade B", 3: "Grade C"}

    if len(pred_classes_list) == 0:
        class_and_confidence = {"No detection found": 0}

    else:
        for idx in range(len(pred_classes_list)):
            grade = grade_dict[pred_classes_list[idx]]
            
            # takes the highest confidence level of each grade
            if grade in class_and_confidence:
                class_and_confidence[grade] = max(
                    class_and_confidence[grade], scores_list[idx])
            else:
                class_and_confidence[grade] = scores_list[idx]

    return [class_and_confidence, out.get_image()]


def yolov8_ebn(img_path, inp_format):
    results = model(img_path)

    xyxys = []
    confidences = []
    class_ids = []

    for result in results:

        boxes = result.boxes.numpy()

        xyxys.append(boxes.xyxy)
        confidences.append(boxes.conf)
        class_ids.append(boxes.cls)

        if (len(boxes.conf) == 0):
            class_and_confidence = {"No detection found": 0}

        else:
            print(boxes.xyxy)
            print("------------------------------------------------------------------")
            print(boxes.conf[0])
            print("------------------------------------------------------------------")
            print(boxes.cls[0])
            print("------------------------------Done--------------------------------")

            grade_name = switch_grade(int(boxes.cls[0]))
            class_and_confidence = {grade_name: boxes.conf[0].item()}
            print(class_and_confidence, '\n')

        # print(im_array)
        im_array = result.plot()  # plot a BGR numpy array of predictions

        if (inp_format == 0):
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        else:
            im = Image.fromarray(im_array[...])

        # im.show()
        im.save('results.jpg')  # save image

    return class_and_confidence


def switch_grade(key):
    key = str(key)
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


def flip(im, a):
    return ["cat", np.flipud(im)]


def image_classify(img, model, inp_format):
    print("Classified model!")
    result = ""
    if model == Model.RESNET50.value:
        # print("Running mobilenet")
        result = resnet50(img)

    elif model == Model.YOLOV8.value:
        # print("Running yolov8")
        result = [yolov8_ebn(img, inp_format), "results.jpg"]
        print("result returned!")

    elif model == Model.DETECTRON.value:
        # print("Running detectron2")
        result = detectron2(img, False)
        
    else:
        print("no value match found")

    return result


def video_classify(img, model, inp_format):
    print("Classified model!")
    result = ""
    if model == Model.RESNET50.value:
        # print("Running mobilenet")
        result = resnet50(img)

    elif model == Model.YOLOV8.value:
        cont = True
        # print("Running yolov8")

        result = [yolov8_ebn(img, inp_format), "results.jpg"]
        print("result returned!")

    elif model == Model.DETECTRON.value:
        # print("Running detectron2")
        result = detectron2(img, True)
        
    else:
        print("no value match found")

    return result


def test(a, b):
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


models = ['resnet50', 'yolov8', 'detectron2', 'mobilenetv2']

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
                image_id = gr.Number(0, visible=False)
            with gr.Column():
                # text_output = gr.Textbox()
                image_output = gr.Label(
                    num_top_classes=3, label="Grade Prediction Output")
                box_output = gr.Image(shape=(56, 56), label="Detected Area")
                clear_image_button = gr.ClearButton(
                    [image_output, box_output])

        # Examples
        gr.Markdown("## EBN Image Examples (Grade A, B, C & multi-EBN image)")
        example_images = [
            ["./sample_images/test_1.jpg", "yolov8"],
            ["./sample_images/test_2.jpg", "yolov8"],
            ["./sample_images/test_3.jpg", "yolov8"],
            ["./sample_images/EBN_comparison.png", "yolov8"]
        ]
        gr.Examples(
            examples=example_images,
            inputs=[image_input, image_model_input]
        )

    with gr.Tab("Real-Time Video Streaming"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Image(source="webcam", streaming=True)
                video_model_input = gr.Dropdown(
                    models, label='Select your Prediction Model')
                video_button = gr.Button("Start Prediction")
                video_id = gr.Number(1, visible=False)
            with gr.Column():
                video_output = gr.Label(
                    num_top_classes=3, label="Grade Prediction Output")
                vid_box_output = gr.Image(shape=(56, 56))
                clear_video_button = gr.ClearButton(
                    [video_output, vid_box_output])

    # image_button.click(test, inputs = text_input, outputs = image_output)    # for testing purposes
    image_button.click(image_classify, inputs=[
                       image_input, image_model_input, image_id], outputs=[image_output, box_output])
    # video_input.change(video_classify, inputs=[
    #                    video_input, video_model_input, video_id], outputs=[video_output, vid_box_output], queue=True, every=5.0)
    video_button.click(video_classify, inputs=[
                       video_input, video_model_input, video_id], outputs=[video_output, vid_box_output])

    gr.Markdown(
        """
        <center><p>Special thanks to Dr Lim Mei Kuan, Dr Chong Chun Yong & Dr Lai Weng Kin</p></center>
        """
    )

# input_image = [gr.Image(type="filepath", label="Upload EBN Image"), gr.Dropdown(models, label='Select your Prediction Model')]
# output_image = [gr.Label(num_top_classes=3, label="Grade Prediction Output"), gr.Image(shape=(56, 56), label="Detected Area")]


if __name__ == "__main__":
    # demo.queue().launch(debug=True)
    demo.launch(debug=True)
