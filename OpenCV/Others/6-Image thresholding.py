import cv2

# Read an image from file
image = cv2.imread('image.jpg')

# Convert image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply simple thresholding
_, thresholded_image = cv2.threshold(grayscale_image, 100, 200, cv2.THRESH_BINARY)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
