import cv2
from collections import defaultdict
import numpy as np
from color_generator import ColorGenerator

class Tracker:
    def __init__(self, model, rtsp_address):
        self.model = model
        self.color_generator = ColorGenerator()
        self.capture = cv2.VideoCapture(rtsp_address)
        self.track_history = defaultdict(lambda: [])
        self.id_colors = {}
        self.timer_start = None
        self.last_known_positions = {}
        self.last_known_position = None
        self.selected_box_color = None
        self.selected_track_id = None
        self.prev_selected_track_id=None
        self.lost_frame_count = 0
        self.max_lost_frames = 30

    def track(self):
        
        while(True):
            success, frame = self.capture.read()
            print(success)
            if success:
                height, width, _ = frame.shape

                frame = cv2.resize(frame, (640, 480))
            
                results = self.model.track(frame, persist=True, classes=0)

                # Set the mouse callback
                cv2.namedWindow('YOLOv8 Inference')
                cv2.setMouseCallback('YOLOv8 Inference', self.mouse_callback, {'results': results})    

                selected_detected = False  # Reset selected_detected flag

                for result in results:
                    for box in result.boxes:
                        confidence = float(box.conf.cpu())
                        if confidence > 0.75:
                            track_id = None
                            if box.id is not None:
                                track_id = int(box.id.cpu().tolist()[0])
                                print("track id: ", track_id)
                                if track_id not in self.last_known_positions:
                                    self.last_known_positions[track_id] = (0, 0, 0, 0)  # Initialize with an empty position

                            if track_id is not None:
                                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                                width = right - left
                                label = results[0].names[int(box.cls)]

                                # Get or generate a unique color for the label
                                if track_id not in self.id_colors:
                                    self.id_colors[track_id] = self.color_generator.generate_color(list(self.id_colors.values()), track_id, avoid_color=(255, 0, 0))
                                
                                color = self.id_colors[track_id]

                                if self.selected_track_id == track_id:
                                    color = self.selected_box_color
                                    self.last_known_position = (left, top, right, bottom)  # Update the last known position
                                    selected_detected = True


                                    if self.timer_start is None:
                                        self.timer_start = cv2.getTickCount()  # Start the timer

                                    timer_now = cv2.getTickCount()  # Get the current tick count
                                    elapsed_time = (timer_now - self.timer_start) / cv2.getTickFrequency()  # Calculate elapsed time in seconds

                                    text_width, text_height = cv2.getTextSize(f"Timer: {elapsed_time:.2f}s", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                                    text_x = left + (width - text_width) // 2
                                    text_y = top - text_height // 2

                                    # Display the timer at the top of the display window
                                    cv2.putText(frame, f"Timer: {elapsed_time:.2f}s", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                cv2.putText(frame, f"{label} {track_id}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(frame, f"confidence: {confidence:.2f}", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # If the selected track ID was not detected, draw the bounding box at the last known position
                if self.selected_track_id is not None and not selected_detected:
                    if self.lost_frame_count < self.max_lost_frames:
                        left, top, right, bottom = self.last_known_position
                        cv2.rectangle(frame, (left, top), (right, bottom), self.selected_box_color, 2)
                        cv2.putText(frame, "Lost", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        self.lost_frame_count += 1
                    else:
                        self.selected_track_id = None  # Reset the selected track ID if the object is lost for too long
                        self.last_known_position = None
        
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            else:
                break
    
    # Mouse callback function
    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # Reset the color of the previously selected box
            if self.prev_selected_track_id is not None:
                existing_colors = list(self.id_colors.values())
                color = self.color_generator.generate_color(existing_colors, self.prev_selected_track_id, avoid_color=(255, 0, 0))
                self.id_colors[self.prev_selected_track_id.cpu().tolist()[0]] = color

            # Check if the click is inside any bounding box
            for result in param['results']:
                for box in result.boxes:
                    
                    left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                    if x >= left and x <= right and y >= top and y <= bottom:
                        print("-============================================================")
                        print("Mouse call back received")
                        print("-============================================================")
                        # Change the color of the bounding box to red
                        self.selected_track_id = box.id
                        self.selected_box_color = (0, 0, 255)  # Red color
                        self.last_known_position = (left, top, right, bottom)  # Update the last known position
                        self.lost_frame_count = 0  # Reset the lost frame count
                        if self.selected_track_id != self.prev_selected_track_id:
                            self.timer_start = None  # Reset the timer
                        self.prev_selected_track_id = self.selected_track_id

                        # Set the color of the selected box to red
                        self.id_colors[self.selected_track_id.cpu().tolist()[0]] = self.selected_box_color

    def release(self):
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
