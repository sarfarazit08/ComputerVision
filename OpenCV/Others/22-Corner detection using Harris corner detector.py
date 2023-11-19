import cv2

image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect corners using the Harris corner detector
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Mark corners on the image
image[corners > 0.01 * corners.max()] = [0, 0, 255]

# Display the image with marked corners
cv2.imshow('Image with Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
