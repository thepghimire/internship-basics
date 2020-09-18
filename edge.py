# Canny Edge Detection 
""" Multi stage algorithm.
- Noise Reduction using 5x5 gaussian filter. 
- finding the intensity gradient of the image
"""
import numpy as np
import cv2 
import matplotlib.pyplot as plt

img = cv2.imread("data/monkeys.jpg")
edges = cv2.Canny(img, 0, 150) #Source, minVal, maxVal

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title("Edges Only"), plt.xticks([]),plt.yticks([])
plt.show()

# cv2.imshow("Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()