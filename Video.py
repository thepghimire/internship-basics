import cv2
import numpy as np

cap = cv2.VideoCapture(0) #First/Primary Webcam in the system

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #Releases the camera.
cv2.destroyAllWindows()

