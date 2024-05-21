import os
from dotenv import load_dotenv
from ultralytics import YOLO
import torch
from tracker import Tracker

load_dotenv(override=True)
rtsp_address = os.getenv('RTSP_ADDRESS')
network_address = os.getenv('NETWORK_ADDRESS')

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

model = YOLO("yolov8n_openvino_model/", task="detect")

with Tracker(model, rtsp_address) as tracker:
    tracker.track()