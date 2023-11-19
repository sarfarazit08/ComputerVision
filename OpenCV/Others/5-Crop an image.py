import cv2

# Read an image from file
image = cv2.imread('image.jpg')

# crop size
cropped_image = image[50:400, 200:500]

# Display the image
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
