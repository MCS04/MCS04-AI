"""
This program contains the Gradio interface of the web application and the means to launch
the web application. It also contains the functions that are calls the deep learning models,
YOLOv8, Detectron2 and MobileNetv2 to predict the EBN grade given an input image. 

Authors: Caleb Tan, Christine Chiong, Wong Yi Zhen Nicholas
Last updated: 1 November 2023.

"""
# Import standard libraries
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

# Import Detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import cv2
import torch
import torchvision

# Import MobileNetv2 utilities
import object_detection
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from matplotlib import pyplot as plt
import operator

# Importing YOLOv8 models
# This line below is required to initialize a custom model
model = YOLO('yolov8n.pt')
model = YOLO('weights/best.pt')
model.fuse()

# Uploading Detectron2 models
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "./weights"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # your number of classes + 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# MobileNetv2 - Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("pipeline.config")
detection_model = model_builder.build(
    model_config=configs['model'], is_training=False)

# MobileNetv2 - Restore checkpoint for mobilenet
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(cfg.OUTPUT_DIR, 'ckpt-6-2')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(
    "label_map.pbtxt")


# Define model names in a Enum class
class Model(Enum):
    YOLOV8 = "yolov8"
    MOBILENET = "mobilenetv2"
    DETECTRON = "detectron2"


# Function for object detection for MobileNetv2
@tf.function
def detect_fn(image):
    """
    Takes in an image and carries out EBN detection within the image. 
    Returns the detections array that contains objects representing 
    detection boxes, score and classes. 

    Parameters:
    image : An image input of a valid format.
    """
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# Function for detection and classification for MobileNetv2
def mobilenet(img, isVideo):
    """
    Processes the input data with the MobileNetv2 model and returns the grade classification result
    with an image annotated with a bounding box representing the EBN detected within.

    Parameters:
    img     : An image input of a valid image format.
    isVideo : A boolean that represents whether the input is from upload image or live.
    """
    print("Running MOBILENET......")
    if not isVideo:
        # read image uploaded
        img = cv2.imread(img)

    else:
        # rgb terbalik for video streaming, no need to cv2.imread as it comes in a numpy array
        img = img[:, :, ::-1]

    # convert image to tensor
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    # detect within the image
    detections = detect_fn(input_tensor)

    # get number of detections and the detections array
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    # define label offser
    label_id_offset = 1

    # make a copy of the image
    image_np_with_detections = image_np.copy()

    # draw bounding box within the copied image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        agnostic_mode=False,
        line_thickness=8,
        min_score_thresh=0)

    # define the class and confidence array dict of each grade
    class_and_confidence = {}
    grade_dict = {0: "Grade A", 1: "Grade B", 2: "Grade C"}

    # if no detection, return no detection found
    if len(detections['detection_classes']) == 0:
        class_and_confidence = {"No detection found": 0}

    else:
        # loop through the detections and get the class and confidence of each grade
        for idx in range(len(detections['detection_classes'])):
            grade = grade_dict[detections['detection_classes'][idx]]

            # takes the highest confidence level of each grade
            if grade in class_and_confidence:
                class_and_confidence[grade] = max(
                    class_and_confidence[grade], detections['detection_scores'][idx])

            else:
                class_and_confidence[grade] = detections['detection_scores'][idx]

    # define figure and axis variables
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot the image and save it to results.jpg
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('results.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    # get the best class and confidence
    best_class_and_confidence = max(
        class_and_confidence.items(), key=operator.itemgetter(1))

    # change to suitable return data type
    return_value = {
        best_class_and_confidence[0]: best_class_and_confidence[1].item()}

    print(return_value)

    return return_value


def detectron2(img, isVideo):
    """
    Processes the input data with the Detectron2 model and returns the grade classification result
    with an image annotated with a bounding box representing the EBN detected within.

    Parameters:
    img     : An image input of a valid image format.
    isVideo : A boolean that represents whether the input is from upload image or live.
    """
    print("Running DETECTRON2......")
    if not isVideo:
        img = cv2.imread(img)

        # Uncomment for testing purposes
        # cv2.imshow("widnow", img)
        # key = cv2.waitKey(10000)
        # if key == 27:
        #     cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

    else:
        # rgb flip for video streaming, no need to cv2.imread as it comes in a numpy array
        img = img[:, :, ::-1]

        # Uncomment for testing purposes
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

    print(class_and_confidence)

    return [class_and_confidence, out.get_image()]


# Function for detection and classification for YOLOv8
def yolov8(img_path, inp_format):
    """
    Processes the input data with the YOLOv8 model and returns the grade classification result
    with an image annotated with a bounding box representing the EBN detected within. 

    Parameters:
    img_path   : An image input of a valid image format.
    inp_format : The input format (upload image or live)
    """
    print("Running YOLOV8......")
    results = model(img_path)

    # get detection and classification results
    for result in results:

        boxes = result.boxes.numpy()

        # Condition for no EBN detected
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

        # plot a BGR numpy array of predictions
        im_array = result.plot()

        # determine the rgb configuration based on input format
        if (inp_format == 0):
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        else:
            im = Image.fromarray(im_array[...])

        # save image
        im.save('results.jpg')

    print(class_and_confidence)

    return class_and_confidence


# YOLOv8 - Function that changes the index to grade name
def switch_grade(key):
    """
    Takes in the index key representing the EBN's grade, returns it in English.  

    Parameters:
    key: The index representing the EBN grade.
    """
    key = str(key)
    if key == "0":
        return "Grade A"
    elif key == "1":
        return "Grade B"
    elif key == "2":
        return "Grade C"


# Function to send input data to correct model in upload interface
def image_classify(img, model, inp_format):
    """
    Takes in input from the Gradio image upload interface and sends it to a model for processing.
    Returns the classfied grade and an image with a bounding box representing the detected EBN within it.

    Parameters:
    img_path   : An image input of a valid image format.
    model      : The selected model.
    inp_format : The input format (upload image or live)
    """
    print(img)
    if img == None:
        raise gr.Error(
            "Please upload an image")

    print("Classifying model......")

    # Model selection
    result = ""
    if model == Model.YOLOV8.value:
        # print("Running yolov8")
        result = [yolov8(img, inp_format), "results.jpg"]
        print("result returned!")

    elif model == Model.DETECTRON.value:
        # print("Running detectron2")
        result = detectron2(img, False)

    elif model == Model.MOBILENET.value:
        result = [mobilenet(img, False), "results.jpg"]

    else:
        print("no value match found")

    return result


# Function to send input data to correct model in live interface
def video_classify(img, model, inp_format):
    """
    Takes in input from the Gradio live interface and sends it to a model for processing.
    Returns the classfied grade and an image with a bounding box representing the detected EBN within it.

    Parameters:
    img_path   : An image input of a valid image format.
    model      : The selected model.
    inp_format : The input format. (upload image or live)
    """
    print("Classifying model......")

    # Model selection
    result = ""
    if model == Model.YOLOV8.value:
        # print("Running yolov8")

        result = [yolov8(img, inp_format), "results.jpg"]
        print("result returned!")

    elif model == Model.DETECTRON.value:
        # print("Running detectron2")
        result = detectron2(img, True)

    elif model == Model.MOBILENET.value:
        result = [mobilenet(img, True), "results.jpg"]

    else:
        print("no value match found")

    return result


# Gradio interface configuration
models = ['yolov8', 'detectron2', 'mobilenetv2']

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <center><h1>Edible Bird's Nest (EBN) Grade Classification System</h1></center>
        <center><p>Created by Caleb Tan, Christine Chiong, Wong Yi Zhen Nicholas & Brian Nge</p></center>
        """
    )

    # Upload image tab interface
    with gr.Tab("Upload Image"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="filepath", label="Upload EBN Image")
                image_model_input = gr.Dropdown(
                    models, label='Select your Prediction Model')
                image_button = gr.Button("Predict")
                image_id = gr.Number(0, visible=False)
            with gr.Column():
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

    # Live tab interface
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

    # Button functions - sends model to model selection and returns output
    image_button.click(image_classify, inputs=[
                       image_input, image_model_input, image_id], outputs=[image_output, box_output])
    video_button.click(video_classify, inputs=[
                       video_input, video_model_input, video_id], outputs=[video_output, vid_box_output])

    gr.Markdown(
        """
        <center><p>Special thanks to Dr Lim Mei Kuan, Dr Chong Chun Yong & Dr Lai Weng Kin</p></center>
        """
    )

if __name__ == "__main__":
    # Launches the Gradio interface
    demo.launch(debug=True)
