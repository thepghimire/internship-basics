import numpy as np
import cv2
import imutils

#Capture via webcam
cap = cv2.VideoCapture(0)

while (1):
    #Read video from webcam
    _, imageFrame = cap.read()
    imageFrame = imutils.resize(imageFrame, width=500)

    
    #Convert imageframe from BGR to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    #Set range for red and define mask:
    red_lower = np.array([136,87,111], np.uint8)
    red_upper = np.array([180,255,255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    #Set range for green and define mask
    green_lower = np.array([25,52,72], np.uint8)
    green_upper = np.array([102,255,255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    
    #Set range for blue and define mask
    blue_lower = np.array([94,80,2], np.uint8)
    blue_upper = np.array([120,255,255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    #morphological Transforms: dilate for each color then bitwise_and operator
    # between imageFrame and mask determines detection of that particular color
    kernel = np.ones((3,3), np.uint8)
    # For red:
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    #For blue
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    #For green
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

    #Create Contour and track the RED color:
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>300:
            x,y,w,h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(imageFrame, "RED COLOR", (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,255))

    #Create contour and track green color:
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x,y,w,h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(imageFrame, "Green Color", (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0))

    #Create Contour and track blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x,y,w,h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(imageFrame, "Blue Color", (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0))

    #Termination
    cv2.imshow("Multiple Color Detection Real time", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
