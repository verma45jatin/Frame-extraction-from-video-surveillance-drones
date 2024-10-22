import cv2
import numpy as np

cap = cv2.VideoCapture("E:\Design Credit Project\los_angeles.mp4")
ret, frame = cap.read()

if ret:
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
else:
    print("Failed to read frame from video")

cap.release()
cv2.destroyAllWindows()
