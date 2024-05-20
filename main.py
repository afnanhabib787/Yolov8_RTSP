import os
from dotenv import load_dotenv
import cv2

load_dotenv()

rtsp_Address = os.getenv('RTSP_ADDRESS')
network_Address = os.getenv('NETWORK_ADDRESS')

capture = cv2.VideoCapture(rtsp_Address)

while(True):
    ret, frame = capture.read()
    print(ret)
    if ret != False:
        cv2.imshow("Live Stream", frame)
    else:
        print("No Frame")
    
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows() 