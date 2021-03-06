import cv2
import numpy as np

path = 'data/monkeys.jpg'
img = cv2.imread(path, cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (150,150), (255,255,255), 15)
cv2.rectangle(img, (15,25), (200,150), (255,255,255), 10)
cv2.circle(img, (100,55), 55, (0,255,0), -1) #-1 will fill it in.

#For polygons 
pts = np.array([[10,5],[20,30],[70,30],[50,10],[100,500]], dtype=np.int32)
pts = pts.reshape((-1,1,2)) #Already the case here. 
cv2.polylines(img, [pts], True, (0,255,255), 3) #True: Connect final point to first point or no?

#Texts on image:
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "Hello There!", (0,130), font, 7, (200,255,255), 5, cv2.LINE_AA)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

