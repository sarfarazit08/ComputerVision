import cv2
import numpy as np

# Load the main image and the template
image = cv2.imread('image.jpg',0)
template = cv2.imread('template.jpg',0)

# Perform template matching
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Set a threshold value to determine a match
threshold = 0.8

# Find locations where the template matches the image above the threshold
locations = np.where(result >= threshold)

# Draw rectangles around the matched areas
for pt in zip(*locations[::1]):
    bottom_right = (pt[0] + template.shape[1], pt[1] + template.shape[0])
    cv2.rectangle(image, pt, bottom_right, (0, 255, 0), 2)

# Display the image with the matched areas
cv2.imshow('Matched Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
