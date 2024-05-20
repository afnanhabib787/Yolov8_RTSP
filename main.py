import os
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import torch
import random

def generate_unique_color(existing_colors, avoid_color=(255, 0, 0)):
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        if color != avoid_color and color not in existing_colors:
            return color


load_dotenv()   
rtsp_Address = os.getenv('RTSP_ADDRESS')
network_Address = os.getenv('NETWORK_ADDRESS')

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

model = YOLO("model/yolov8n.pt") 

capture = cv2.VideoCapture(rtsp_Address)

label_colors = {}

while(True):
    success, frame = capture.read()
    if success:
        height, width, _ = frame.shape

        frame = cv2.resize(frame, (640, 480))
        
        results = model.track(frame, persist=True, classes=0)
        
        for result in results:
            for box in result.boxes:
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                width = right - left
                height = bottom - top
                center = (left + int((right-left)/2), top + int((bottom-top)/2))
                label = results[0].names[int(box.cls)]
                confidence = float(box.conf.cpu())
                if label not in label_colors:
                    label_colors[label] = generate_unique_color(label_colors.values())

                color = label_colors[label]

                cv2.rectangle(frame, (left, top),(right, bottom), color, 2)
                cv2.putText(frame, label,(left, top),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        
    
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    else:
        break
    
capture.release()
cv2.destroyAllWindows() 
