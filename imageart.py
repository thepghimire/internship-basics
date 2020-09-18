import cv2
import numpy as np

img1 = cv2.imread('data/monkeys.jpg')
img2 = cv2.imread('data/morphling.png')

img = cv2.add(img1 + img2)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()