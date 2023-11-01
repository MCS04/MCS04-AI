# import pytest
import unittest
from ultralytics import YOLO

class TestAccuracy(unittest.TestCase):
    def setUp(self):
        self.model = YOLO('best.pt')
    
    def testAccuracyNoEbn(self):
        results = self.model.predict("notebn.jpg")
        self.assertEqual(results[0].probs, None)        
    
    def testAccuracyAEbn(self):
        results = self.model.predict("A.bmp")
        accuracy = 0
        for result in results:
            
            box = result.boxes.numpy()
            
            if box.cls == 0:
                accuracy = max(accuracy, box.conf[0])
        
        self.assertGreaterEqual(accuracy, 0.8)
        
    def testAccuracyBEbn(self):
        results = self.model.predict("B.bmp")
        accuracy = 0
        for result in results:
            
            box = result.boxes.numpy()
            
            if box.cls == 1:
                accuracy = max(accuracy, box.conf[0])
        
        self.assertGreaterEqual(accuracy, 0.8)        
        
    def testAccuracyCEbn(self):
        results = self.model.predict("C.bmp")
        accuracy = 0
        for result in results:
            
            box = result.boxes.numpy()
            index = -1
            
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 2:
                    index = ebnclass
                                
            accuracy = max(accuracy, box.conf[index])
        
        self.assertGreaterEqual(accuracy, 0.8)  
        
    def testAccuracyMultipleAEbn(self):
        results = self.model.predict("multipleA.jpg")
        accuracy = []
        for result in results:
            
            box = result.boxes.numpy()
            
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 0:
                    accuracy.append(box.conf[ebnclass])
        
        if len(accuracy) == 0:
            accuracy.append(0)
            accuracy.append(0)
        
        elif len(accuracy) == 1:
            accuracy.append(0)
            
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)   
    
    def testAccuracyMultipleBEbn(self):
        results = self.model.predict("multipleB.jpg")
        accuracy = []
        for result in results:
            
            box = result.boxes.numpy()
            
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 1:
                    accuracy.append(box.conf[ebnclass])
        
        if len(accuracy) == 0:
            accuracy.append(0)
            accuracy.append(0)
        
        elif len(accuracy) == 1:
            accuracy.append(0)
                                
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)   
    
    def testAccuracyMultipleCEbn(self):
        results = self.model.predict("multipleC.jpg")
        accuracy = []
        for result in results:
            
            box = result.boxes.numpy()
            
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 2:
                    accuracy.append(box.conf[ebnclass])
        
        if len(accuracy) == 0:
            accuracy.append(0)
            accuracy.append(0)
        
        elif len(accuracy) == 1:
            accuracy.append(0)
            
        self.assertGreaterEqual(accuracy[0], 0.8)
        self.assertGreaterEqual(accuracy[1], 0.8)   
    
    def testAccuracyMultipleAandBEbn(self):
        results = self.model.predict("multipleAandB.jpg")
        accuracy = [0, 0]
        for result in results:
            
            box = result.boxes.numpy()
            
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 0:
                    accuracy[0] = max(accuracy[0], box.conf[ebnclass])
                elif box.cls[ebnclass] == 1:
                    accuracy[1] = max(accuracy[1], box.conf[ebnclass])
                                
        self.assertGreaterEqual(accuracy[0], 0.8, "A not reached satisfactory accuracy")
        self.assertGreaterEqual(accuracy[1], 0.8, "B not reached satisfactory accuracy")         
    
    def testAccuracyMultipleAandCEbn(self):
        results = self.model.predict("multipleAandC.jpg")
        accuracy = [0, 0]
        for result in results:
            
            box = result.boxes.numpy()
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 0:
                    accuracy[0] = max(accuracy[0], box.conf[ebnclass])
                elif box.cls[ebnclass] == 2:
                    accuracy[1] = max(accuracy[1], box.conf[ebnclass])
                                
        self.assertGreaterEqual(accuracy[0], 0.8, "A not reached satisfactory accuracy")
        self.assertGreaterEqual(accuracy[1], 0.8, "C not reached satisfactory accuracy")   
        
    def testAccuracyMultipleBandCEbn(self):
        results = self.model.predict("multipleBandC.jpg")
        accuracy = [0, 0]
        for result in results:
            
            box = result.boxes.numpy()
            
            for ebnclass in range(len(box.cls)):
                if box.cls[ebnclass] == 1:
                    accuracy[0] = max(accuracy[0], box.conf[ebnclass])
                elif box.cls[ebnclass] == 2:
                    accuracy[1] = max(accuracy[1], box.conf[ebnclass])
                                
        self.assertGreaterEqual(accuracy[0], 0.8, "B not reached satisfactory accuracy")
        self.assertGreaterEqual(accuracy[1], 0.8, "C not reached satisfactory accuracy")   
    
if __name__ == "__main__":
    unittest.main()