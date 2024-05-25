import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
import base64
import threading
import requests
import json

# Raspberry PI IP address
MQTT_BROKER = "103.127.97.247"
# Topic on which frame will be published
TOPIC_KIRIM_GAMBAR = "home/server/9218291212"
TOPIC_TERIMA_PESAN = "home/terima/9218291212"

client = mqtt.Client()
client.connect(MQTT_BROKER)

img_url = 'https://ajang.avisha.id/wp-content/uploads/2024/05/image-1.png'  # URL gambar yang ingin dikirim
token = "6992326041:AAGLgeu8d8r-3YAD4PeLM6Zfh287l6Ws4nw"  # Token bot Telegram
chat_id = "262249300"  # ID chat di Telegram
caption = "People Detected!!! "

# Deklarasi variabel global
global person_detected_flag
person_detected_flag = False

global num_seconds
num_seconds = 500  # Set initial countdown time

def countdown_timer():
    global num_seconds
    while num_seconds > 0:
        # print(num_seconds)
        num_seconds -= 1
        time.sleep(1)
    # print("Countdown reached 0.")

# Fungsi send_telegram_photo_from_frame yang telah kita buat sebelumnya
def send_telegram_photo_from_frame(frame, token, chat_id, caption):
    global num_person
    global person_detected_flag
    global num_seconds
    
    if num_person >= 1:
        num_seconds = 100
        if not person_detected_flag:
            person_detected_flag = True
            print("Ada Orang")
            url = f'https://api.telegram.org/bot{token}/sendPhoto'
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'photo': img_encoded.tobytes()}
            params = {'chat_id': chat_id, 'caption': caption}
            resp = requests.post(url, params=params, files=files)
            print(f'Response Code: {resp.status_code}')
            
            url = "http://192.168.0.103/rfid-doorlock/input_notif.php"
            params = {'chat_id': chat_id, 'caption': caption}
            resp = requests.post(url, params=params, files=files)
            print(f'Response Code: {resp.status_code}')
            
    if num_person <= 0 and person_detected_flag:
        if num_seconds <= 0:
            print("Tidak Ada Orang")
            person_detected_flag = False

def publish_frame(image):
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    client.publish(TOPIC_KIRIM_GAMBAR, jpg_as_text)
    # print("Data published.")

def read_classes(classesPath):
    with open(classesPath, 'r') as f:
        classes = f.read().splitlines()

    classes.insert(0, '__background__')
    colorList = np.random.uniform(0, 255, size=(len(classes), 3))
    return classes, colorList

def send_frame_to_telegram(image):
    send_telegram_photo_from_frame(image, token, chat_id, caption)

def detect_objects(videoPath, configPath, modelPath, classes, colorList):
    global person_detected_flag  # Declare as global
    net = cv2.dnn_DetectionModel(modelPath, configPath)
    net.setInputSize(224, 224)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    global num_person
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
                # classColor = [int(c) for c in colorList[classLabelID]]
                classColor = (0, 255, 0)

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
        
        telegram_thread = threading.Thread(target=send_frame_to_telegram, args=(image,))
        telegram_thread.start()
        
        timer_thread = threading.Thread(target=countdown_timer)
        timer_thread.start()

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