import cv2
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('face_deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load the input image
image = cv2.imread('image2.jpg')

# Preprocess the image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Set the input blob for the network
net.setInput(blob)

# Forward pass through the network
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        # Extract the bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        (startX, startY, endX, endY) = box.astype(int)

        # Draw the bounding box and confidence
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)
        text = f'{confidence * 100:.2f}%'
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)

# Display the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
