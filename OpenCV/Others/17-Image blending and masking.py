import cv2
import numpy as np

image1 = cv2.imread('image.jpg')
image2 = cv2.imread('mask.jpg')

# Resize image2 to match image1 dimensions
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Create a mask
mask = np.zeros_like(image1)
mask[50:400, 200:500] = 255

# Blend the images using the mask
blended_image = cv2.addWeighted(image1, 0.7, image2_resized, 0.3, 0)
masked_image = cv2.bitwise_and(image1, mask)

# Display the images
cv2.imshow('Blended Image', blended_image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
