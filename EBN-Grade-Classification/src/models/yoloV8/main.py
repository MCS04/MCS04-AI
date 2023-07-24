# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("yolov8n.yaml") # build a new model from scratch
#
# ## Don't use this first
# # Use the model
# results = model.train(data="config.yaml", epochs=1)  # train the model

from roboflow import Roboflow
import os
import json

from roboflow import Roboflow
rf = Roboflow(api_key="SuklAeRTbtD1k4yDlJsC")

workspace = rf.workspace("ebn")

# replace with the IDs of your projects
# You can retrieve the project IDs from the Roboflow Dashboard by
# taking the last part of each project URL
# (i.e. https://app.roboflow.com/test/123) would have the project ID "123"
# in the workspace "test"

projects = ["bird-nest-exr6l"]

def generate_and_train(project: str, configuration: dict) -> None:
    """
    Generate a version of a model and commence training for the new version.
    """
    rf_project = workspace.project(project)

    version_number = rf_project.generate_version(configuration)

    project_item = workspace.project(project).version(version_number)

    project_item.train()


with open("starter.json","r") as f:
    configuration = json.load(f)

generate_and_train(projects[0], configuration)
def apply_multiple_experiments(project: str) -> None:
    """
    For each configuration in the "configurations" folder,
    generate and train a modelfor the specified project.
    """
    for configuration in os.listdir("configurations"):
        with open(f"configurations/{configuration}") as f:
            configuration = json.load(f)

        generate_and_train(project, configuration)

for project in projects:
    apply_multiple_experiments(project)