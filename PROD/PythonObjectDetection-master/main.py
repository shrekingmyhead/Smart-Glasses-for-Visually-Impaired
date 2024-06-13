from detector import *
import os 

def main():
    #Set Videos Option
    videoPath = "test_videos/sample_4.mp4"
    #Web Cam Option
    #videoPath = 0
    configPath = os.path.join("model_data", "ssd_mobile_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.startVideo()

if __name__ == "__main__":
    main()

