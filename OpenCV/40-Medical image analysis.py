import cv2
import SimpleITK as sitk

# Load the brain MRI image using SimpleITK
image = sitk.ReadImage("D:\Projects\Python\MD5\\02ace403d8b1533289b1da30a6b02bed")

# Convert the SimpleITK image to a NumPy array
array = sitk.GetArrayFromImage(image)

# Preprocess the image (e.g., normalization, denoising, etc.)

# Perform tumor detection using OpenCV
_, binary = cv2.threshold(array, 100, 255, cv2.THRESH_BINARY)

# Perform morphological operations (e.g., erosion, dilation) for better segmentation

# Find contours of the detected tumor regions
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours and draw bounding boxes
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(array, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image with detected tumor regions
cv2.imshow('Tumor Detection', array)
cv2.waitKey(0)
cv2.destroyAllWindows()
