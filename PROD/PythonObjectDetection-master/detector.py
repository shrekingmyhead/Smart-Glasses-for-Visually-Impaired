import cv2 
import numpy as np 
import time 

np.random.seed(5)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath 
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        #Initialize the network and set parameter.
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as file:
            self.classesList = file.read().splitlines()

        self.classesList.insert(0, "__Background__")

        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3) )

    def startVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        if(cap.isOpened() == False):
            print("Error Opening The Camera")
            return 
        (succes, image) = cap.read()
        while(succes):
            classLabelIds, confidence_score, bboxs = self.net.detect(image, confThreshold=0.5)

            bboxs = list(bboxs)
            
            confidence_score = list(np.array(confidence_score).reshape(1, -1)[0])
            confidence_score = list(map(float, confidence_score))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidence_score, score_threshold=0.5, nms_threshold=0.2)

            if(len(bboxIdx) != 0):
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[ np.squeeze(bboxIdx[i]) ]
                    
                    classConfidence = confidence_score[ np.squeeze(bboxIdx[i]) ]
                    classLabelID = np.squeeze( classLabelIds[ np.squeeze(bboxIdx[i]) ] )
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    on_screen_text = "{}:{:.2f}".format(classLabel, classConfidence)
        
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=3)
                    cv2.putText(image, on_screen_text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, classColor)

            cv2.imshow("Result", image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (succes, image) = cap.read()

        cv2.destroyAllWindows()

