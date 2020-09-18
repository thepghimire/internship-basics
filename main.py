import cv2
import numpy as np

path = "data/morphling.png"
image = cv2.imread(path, 0)

# Working with Morphological Transforms. 

kernel = np.ones((5,5), np.uint8)
# Dilation: pixel value 1 if any of the pixels of kernel is 1.
dilation = cv2.dilate(image, kernel, iterations=1)
cv2.imshow("Dilation", dilation)

#Erosion: opposite of dilation. pixel value 1 only if ALL pixels under kernel is 1
# Helps in removing white noise. 
# All pixels near boundary will be discarded (depending on kernel size)
# and thickness of foreground object will also decrease.
erosion = cv2.erode(image, kernel, iterations=1)
cv2.imshow("Erosion", erosion)
cv2.imshow("Original", image)

#Opening: Erosion followed by dilation; helps in removing noise. 
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening", opening)

#Closing: Dilation followed by Erosion: useful in filling small holes inside foreground objects
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing", closing)

#Morphological Gradient: Difference between dilation and erosion of an image
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("Morphological Gradient", gradient)

#Top Hat: difference between input image and opening of image
top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Tophat", top_hat)

#Black Hat: difference between Closing and Input image
black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("Black Hat", black_hat)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break