# import pytest
import unittest
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import cv2


class TestAccuracy(unittest.TestCase):

    def preprocess(self, img):
        img = cv2.imread(img)

        outputs = self.model(img)

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

        if len(pred_classes_list) == 0:
            class_and_confidence = {"No detection found": 0}

        else:
            for idx in range(len(pred_classes_list)):
                grade = pred_classes_list[idx]

                # takes the highest confidence level of each grade
                if grade in class_and_confidence:
                    class_and_confidence[grade] = max(
                        class_and_confidence[grade], scores_list[idx])
                else:
                    class_and_confidence[grade] = scores_list[idx]

        return class_and_confidence

    def multiplePreprocess(self, img):
        img = cv2.imread(img)

        outputs = self.model(img)

        v = Visualizer(img[:, :, ::-1],
                       scale=0.8
                       )
        class_and_confidence = []

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # getting the predicted classes
        pred_classes = outputs["instances"].pred_classes
        pred_classes_list = pred_classes.cpu().numpy().tolist()

        # getting the predicted scores
        scores = outputs["instances"].scores
        scores_list = scores.cpu().numpy().tolist()

        if len(pred_classes_list) == 0:
            class_and_confidence = {"No detection found": 0}

        else:
            for idx in range(len(pred_classes_list)):
                grade = pred_classes_list[idx]
                class_and_confidence.append((grade, scores_list[idx]))

        return class_and_confidence

    def checkForClass(self, classes_conf, grade):
        top_conf = []
        for idx, class_conf in enumerate(classes_conf):
            if class_conf[0] == grade:
                top_conf.append(class_conf[1])
        top_conf.sort(reverse=True)

        if len(top_conf) == 0:
            top_conf.append(0)
            top_conf.append(0)

        elif len(top_conf) == 1:
            top_conf.append(0)

        return [top_conf[0], top_conf[1]]

    def checkForMultipleClass(self, classes_conf, grade1, grade2):
        top_conf = {}
        for idx, class_conf in enumerate(classes_conf):
            top_conf[class_conf[0]] = class_conf[1]
        
        if grade1 not in top_conf:
            top_conf[grade1] = 0
        if grade2 not in top_conf:
            top_conf[grade2] = 0
        
        return [top_conf[grade1], top_conf[grade2]]

    def setUp(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.OUTPUT_DIR = "./"
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # your number of classes + 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)
        self.model = predictor

    def testAccuracyNoEbn(self):
        class_and_confidence = self.preprocess("notebn.jpg")
        self.assertEqual(len(class_and_confidence), 0)

    def testAccuracyAEbn(self):
        class_and_confidence = self.preprocess("A.jpg")
        self.assertGreaterEqual(class_and_confidence[1], 0.8)

    def testAccuracyBEbn(self):
        class_and_confidence = self.preprocess("B.jpg")
        self.assertGreaterEqual(class_and_confidence[2], 0.8)

    def testAccuracyCEbn(self):
        class_and_confidence = self.preprocess("C.jpg")
        self.assertGreaterEqual(class_and_confidence[3], 0.8)

    def testAccuracyMultipleAEbn(self):
        class_and_confidence = self.multiplePreprocess("multipleA.jpg")
        accuracy = self.checkForClass(class_and_confidence, 1)
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)

    def testAccuracyMultipleBEbn(self):
        class_and_confidence = self.multiplePreprocess("multipleB.jpg")
        accuracy = self.checkForClass(class_and_confidence, 2)
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)

    def testAccuracyMultipleCEbn(self):
        class_and_confidence = self.multiplePreprocess("multipleC.jpg")
        accuracy = self.checkForClass(class_and_confidence, 3)
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)

    def testAccuracyMultipleAandBEbn(self):
        class_and_confidence = self.multiplePreprocess("multipleAandB.jpg")
        accuracy = self.checkForMultipleClass(class_and_confidence, 1, 2)
        self.assertGreaterEqual(
            accuracy[0], 0.8, "A not reached satisfactory accuracy")
        self.assertGreaterEqual(
            accuracy[1], 0.8, "B not reached satisfactory accuracy")

    def testAccuracyMultipleAandCEbn(self):
        class_and_confidence = self.multiplePreprocess("multipleAandC.jpg")
        accuracy = self.checkForMultipleClass(class_and_confidence, 1, 3)
        self.assertGreaterEqual(
            accuracy[0], 0.8, "A not reached satisfactory accuracy")
        self.assertGreaterEqual(
            accuracy[1], 0.8, "C not reached satisfactory accuracy")

    def testAccuracyMultipleBandCEbn(self):
        class_and_confidence = self.multiplePreprocess("multipleBandC.jpg")
        accuracy = self.checkForMultipleClass(class_and_confidence, 2, 3)
        self.assertGreaterEqual(
            accuracy[0], 0.8, "B not reached satisfactory accuracy")
        self.assertGreaterEqual(
            accuracy[1], 0.8, "C not reached satisfactory accuracy")


if __name__ == "__main__":
    unittest.main()
