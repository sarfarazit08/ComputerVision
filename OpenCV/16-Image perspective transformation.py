import cv2
import numpy as np

image = cv2.imread('image.jpg')
height, width = image.shape[:2]

# Define source and destination points
source_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
destination_points = np.float32([[0, 0], [width - 1, 0], [int(0.3 * width), height - 1], [int(0.7 * width), height - 1]])

# Calculate perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

# Apply perspective transformation
transformed_image = cv2.warpPerspective(image, perspective_matrix, (width, height))

# Display the original and transformed image
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
