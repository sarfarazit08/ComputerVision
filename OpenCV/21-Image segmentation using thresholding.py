import cv2

image = cv2.imread('image.jpg', 0)

# Apply thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Display the original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
