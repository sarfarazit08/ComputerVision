import cv2

# Read an image from file
image = cv2.imread('image.jpg')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply simple thresholding
_, thresholded_image = cv2.threshold(grayscale_image, 30, 100, cv2.THRESH_BINARY)
# Display the simple thresholded Image
cv2.imshow('Simple Thresholded Image', thresholded_image)


# Apply adaptive thresholding
adaptive_thresholded_image = cv2.adaptiveThreshold(grayscale_image, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# Display the Adaptive Thresholded Image
cv2.imshow('Adaptive Thresholded Image', adaptive_thresholded_image)


# Apply Otsu's thresholding
_, otsu_thresholded_image = cv2.threshold(grayscale_image, 0, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Display the Otsu's Thresholded Image
cv2.imshow("Otsu's Thresholded Image", otsu_thresholded_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
