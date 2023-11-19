import cv2

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Image with Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
