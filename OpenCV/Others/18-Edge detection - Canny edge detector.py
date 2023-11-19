import cv2

image = cv2.imread('image.jpg', 0)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 200)

# Display the original image and the detected edges
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
