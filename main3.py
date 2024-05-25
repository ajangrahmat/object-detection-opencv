import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
import base64
import threading

# Raspberry PI IP address
MQTT_BROKER = "103.127.97.247"
# Topic on which frame will be published
MQTT_SEND = "home/server/9218291212"

client = mqtt.Client()
client.connect(MQTT_BROKER)

def publish_frame(image):
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    client.publish(MQTT_SEND, jpg_as_text)
    print("Data published.")

def read_classes(classesPath):
    with open(classesPath, 'r') as f:
        classes = f.read().splitlines()

    classes.insert(0, '__background__')
    colorList = np.random.uniform(0, 255, size=(len(classes), 3))
    return classes, colorList

def detect_objects(videoPath, configPath, modelPath, classes, colorList):
    net = cv2.dnn_DetectionModel(modelPath, configPath)
    net.setInputSize(224, 224)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    num_person = 0
    prev_time = time.time()

    while True:
        success, image = cap.read()
        if not success:
            break

        classLabelIDs, confidences, bboxs = net.detect(image, confThreshold=0.4, nmsThreshold=0.2)
        bboxs = list(bboxs)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, 0.5, 0.2)

        num_person = 0
        if len(bboxIdx) != 0:
            for i in range(0, len(bboxIdx)):
                bbox = bboxs[np.squeeze(bboxIdx[i])]
                classConfidence = confidences[np.squeeze(bboxIdx[i])]
                classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                classLabel = classes[classLabelID]
                classColor = [int(c) for c in colorList[classLabelID]]

                if classLabel == 'person':
                    num_person += 1

                displayText = '{}: {:.2f}%'.format(classLabel, classConfidence * 100)

                x, y, w, h = bbox
                cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), classColor, 2)
                cv2.putText(image, displayText, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, classColor, 2)

                lineWidth = int(round((w+h)*0.005))
                cv2.line(image, (x,y), (x+lineWidth,y), classColor, 5)
                cv2.line(image, (x,y), (x,y+lineWidth), classColor, 5)
                cv2.line(image, (x+w,y), (x+w-lineWidth,y), classColor, 5)
                cv2.line(image, (x+w,y), (x+w,y+lineWidth), classColor, 5)

        # Calculate FPS
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, f"Persons: {num_person}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Output", image)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

        # Create a thread to publish the frame
        publish_thread = threading.Thread(target=publish_frame, args=(image,))
        publish_thread.start()

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    videoPath = 1  # Specify your video file path
    configPath = 'model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    modelPath = 'model_data/frozen_inference_graph.pb'
    classesPath = 'model_data/coco.names'

    classes, colorList = read_classes(classesPath)
    detect_objects(videoPath, configPath, modelPath, classes, colorList)

if __name__ == "__main__":
    main()
