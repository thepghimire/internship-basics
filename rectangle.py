import cv2
import numpy as np

img = cv2.imread("data/rectangle.png")
img = cv2.medianBlur(img, 5)
# kernel = np.ones((3,3), np.uint8)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# openingimg = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)


# COLOR_MIN = np.array([20,80,80], np.uint8) #Yellow ko darkest shade
# COLOR_MAX = np.array([40,255,255], np.uint8) #Yellow ko lightest shade
COLOR_MIN = np.array([20,20,80], np.uint8) #Red ko dark shade
COLOR_MAX = np.array([60,60,255], np.uint8) #red ko lightest shade
frame_threshed = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)

imgray = frame_threshed
ret, thresh = cv2.threshold(frame_threshed, 150,255,0)
cv2.imshow("Thresh", thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()