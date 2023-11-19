import cv2

# Read an image from file
image = cv2.imread('image.jpg')

# Get the size of the image
size = image.size

# Get the shape of the image
shape = image.shape
print('Image Size:', size)
print('Image Shape:', shape)

# Resize the image to a specific width and height (increment by 200 pixels each side)
resized_image = cv2.resize(image, (image.shape[0] - 200, image.shape[1]- 50))

# Get the size of the image
size = resized_image.size

# Get the shape of the image
shape = resized_image.shape

print('Resized Image Size:', size)
print('Resized Image Shape:', shape)

# Display the Resized image
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

