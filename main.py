from Detector import *
import os

def main():
    videoPath = "rtsp://admin:1234567890@192.168.0.140:8554/sub" 
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")
    
    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()
    
if __name__ == "__main__":
    main()
