# YOLOv8 Object Tracker

This project is an implementation of an person tracker using the YOLOv8 model. The tracker processes video frames from an RTSP stream, detects person, and tracks them using unique colors for each object. Additionally, it allows users to select objects by clicking on them, which changes their bounding box color to red and displays a timer showing how long the object has been selected.

## Features

- Person detection and tracking using YOLOv8.
- Unique colors for each tracked object.
- Clickable bounding boxes to select objects.
- Timer display for selected objects.
- Handling lost objects by keeping track of the last known position for a certain number of frames.

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- torch

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/afnanhabib787/Yolov8_RTSP.git
    cd Yolov8_RTSP
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have YOLOv8 model and related files in your project directory or adjust the code to load the model from your desired location.

## Usage

```
python main.py
```