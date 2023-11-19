import cv2
import numpy as np

# Read an image from file
image = cv2.imread('image.jpg')

# Apply Gaussian blur to the image
blurred_image = cv2.GaussianBlur(image, (13, 13), 0)
# Display the Blurred Image
cv2.imshow('Blurred Image', blurred_image)

# Apply sharpening filter to the image
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, kernel)

# Display the Sharpened Image
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

