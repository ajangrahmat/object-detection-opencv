import cv2
import numpy as np
import time
from flask import Flask, render_template, Response

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        
        self.classes = None
        self.colorList = None
        self.num_person = 0
        self.prev_time = time.time()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classes = f.read().splitlines()

        self.classes.insert(0, '__background__')
        self.colorList = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        self.readClasses()

        while True:
            success, image = cap.read()
            if not success:
                break
            
            classLabelIDs, confidences, bboxs = self.net.detect(
                image, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, 0.5, 0.2)

            self.num_person = 0
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(
                        classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classes[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    
                    if classLabel == 'person':
                        self.num_person += 1
                    
                    displayText = '{}: {:.2f}%'.format(classLabel, classConfidence * 100)
                        
                    x, y, w, h = bbox

                    cv2.rectangle(image, (int(x), int(y)),
                                (int(x+w), int(y+h)), classColor, 2)
                    cv2.putText(image, displayText, (int(x), int(y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, classColor, 2)
                    
                    lineWidth = int(round((w+h)*0.005))
                    
                    cv2.line(image, (x,y), (x+lineWidth,y), classColor, 5)
                    cv2.line (image, (x,y), (x,y+lineWidth), classColor, 5)
                    cv2.line(image, (x+w,y), (x+w-lineWidth,y), classColor, 5)
                    cv2.line(image, (x+w,y), (x+w,y+lineWidth), classColor, 5)
            
            # Calculate FPS
            cur_time = time.time()
            fps = 1 / (cur_time - self.prev_time)
            self.prev_time = cur_time
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Persons: {self.num_person}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Output", image)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()