import cv2
import numpy as np

img = cv2.imread("data/morphling.png")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# The above line of code only finds the contours. We need to draw it.
# Drawing the contours in the main image:

# cv2.drawContours(img, contours, -1, (0,255,0), 1)
# cv2.imshow("Image", img)

# Moments; used to calculate features like center of mass, area, etc.
cnt = contours[0]
M = cv2.moments(cnt)
print("Dictionary M:\n",M)
# print("CNT::::\n", cnt)
# Centroid calculated as: m10/m00, m01/m00
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#Contour Area: given by cv2.contourArea() OR from M['m00']
area = cv2.contourArea(cnt)
print("Contour Area: \n", area)

#Contour perimeter: Also "arc Length", cv2.arcLength(). 
perimeter = cv2.arcLength(cnt, True)
print("Perimeter is:\n", perimeter)

#Contour approximation: Approximates contour shape to another shape with less no of
# Vertices, depending on precision we specify. 
epsilon = 0.8*cv2.arcLength(cnt, False)
approx = cv2.approxPolyDP(cnt, epsilon, False)
print("\n Approx:", approx)
cv2.drawContours(img, approx, -1, (00,0,255),3)
# cv2.imshow("image2", img)

#Bounding Rectangle:
#Straight:
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
# cv2.imshow("Image", img)

#Rectangle with min area (rotated rectangle)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, (255,0,0), 2)
# cv2.imshow("Image3", img)

#minimum enclosing circle
(x,y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, center, radius, (0,255,0), 2)
# cv2.imshow("Circled", img)

# Fitting and ellipse
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img, ellipse, (255,255,0), 2)
cv2.imshow("Image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
