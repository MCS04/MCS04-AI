# import pytest
import unittest
import object_detection
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from matplotlib import pyplot as plt
import operator
import tensorflow as tf
from tensorflow import keras
from roboflow import Roboflow
import numpy as np
import os
import cv2


configs = config_util.get_configs_from_pipeline_file("pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

class TestAccuracy(unittest.TestCase):
    
    def setUp(self):

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join('./weights/', 'ckpt-6-2')).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt")
        
    
    def prediction(self, img, isVideo):
        if not isVideo:
            img = cv2.imread(img)

        else:
            # rgb terbalik for video streaming, no need to cv2.imread as it comes in a numpy array
            img = img[:, :, ::-1]

        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=1,
                    agnostic_mode=False,
                    line_thickness=8,
                    min_score_thresh=0)
        
        class_and_confidence = {}
        grade_dict = {0: "Grade A", 1: "Grade B", 2: "Grade C"}
            
        if len(detections['detection_classes']) == 0:
            class_and_confidence = {"No detection found": 0}

        else:
            for idx in range(len(detections['detection_classes'])):
                grade = grade_dict[detections['detection_classes'][idx]]

                # takes the highest confidence level of each grade
                if grade in class_and_confidence:
                    class_and_confidence[grade] = max(
                        class_and_confidence[grade], detections['detection_scores'][idx])
                    
                else:
                    class_and_confidence[grade] = detections['detection_scores'][idx]
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('results.jpg', bbox_inches='tight', pad_inches = 0, dpi = 300)

        best_class_and_confidence = max(class_and_confidence.items(), key=operator.itemgetter(1))

        return_value = {best_class_and_confidence[0] : best_class_and_confidence[1].item()}        

        return return_value
        
    def testAccuracyNoEbn(self):
        class_and_confidence = self.prediction("notebn.jpg", False)
        self.assertEqual(len(class_and_confidence), 0)

    def testAccuracyAEbn(self):
        class_and_confidence = self.prediction("A.jpg", False)
        self.assertGreaterEqual(class_and_confidence[1], 0.8)

    def testAccuracyBEbn(self):
        class_and_confidence = self.prediction("B.jpg", False)
        self.assertGreaterEqual(class_and_confidence[2], 0.8)

    def testAccuracyCEbn(self):
        class_and_confidence = self.prediction("C.jpg", False)
        self.assertGreaterEqual(class_and_confidence[3], 0.8)

    def testAccuracyMultipleAEbn(self):
        class_and_confidence = self.prediction("multipleA.jpg", False)
        accuracy = self.checkForClass(class_and_confidence, 1)
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)

    def testAccuracyMultipleBEbn(self):
        class_and_confidence = self.prediction("multipleB.jpg", False)
        accuracy = self.checkForClass(class_and_confidence, 2)
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)

    def testAccuracyMultipleCEbn(self):
        class_and_confidence = self.prediction("multipleC.jpg", False)
        accuracy = self.checkForClass(class_and_confidence, 3)
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)

    def testAccuracyMultipleAandBEbn(self):
        class_and_confidence = self.prediction("multipleAandB.jpg", False)
        accuracy = self.checkForMultipleClass(class_and_confidence, 1, 2)
        self.assertGreaterEqual(
            accuracy[0], 0.8, "A not reached satisfactory accuracy")
        self.assertGreaterEqual(
            accuracy[1], 0.8, "B not reached satisfactory accuracy")

    def testAccuracyMultipleAandCEbn(self):
        class_and_confidence = self.prediction("multipleAandC.jpg", False)
        accuracy = self.checkForMultipleClass(class_and_confidence, 1, 3)
        self.assertGreaterEqual(
            accuracy[0], 0.8, "A not reached satisfactory accuracy")
        self.assertGreaterEqual(
            accuracy[1], 0.8, "C not reached satisfactory accuracy")

    def testAccuracyMultipleBandCEbn(self):
        class_and_confidence = self.prediction("multipleBandC.jpg", False)
        accuracy = self.checkForMultipleClass(class_and_confidence, 2, 3)
        self.assertGreaterEqual(
            accuracy[0], 0.8, "B not reached satisfactory accuracy")
        self.assertGreaterEqual(
            accuracy[1], 0.8, "C not reached satisfactory accuracy")


if __name__ == "__main__":
    unittest.main()
