import numpy as np
import cv2

img = cv2.imread('data/monkeys.jpg', cv2.IMREAD_COLOR)

#Location (Color values of this location)
img[55,55] = [255,255,255] #changed it to 255,255,255
px = img[55,55] 

print(px)

# ROI (Region within the image)
img[100:150, 100:150] = [255,255,255]

monkey_face = img[300:550, 109:292]
img[0:250, 0:183] = monkey_face


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()