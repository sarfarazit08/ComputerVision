import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', 0)

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display the original and equalized images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(equalized_image, cmap='gray')
axes[1].set_title('Equalized Image')
axes[1].axis('off')
plt.show()
