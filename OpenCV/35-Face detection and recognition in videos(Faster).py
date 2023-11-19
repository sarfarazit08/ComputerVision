import cv2
import numpy as np

# Load the pre-trained face detection model
prototxt = 'face_deploy.prototxt'
weights = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, weights)

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if not ret:
        break

    # Perform face detection using the pre-trained model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Iterate over the detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Threshold for detection confidence
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype(int)

            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
