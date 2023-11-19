import cv2

# Load the pre-trained models for object detection and face detection
object_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('image2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect objects (cats) in the image
objects = object_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Detect faces in the image
faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw bounding boxes around the detected objects and faces
for (x, y, w, h) in objects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv2.putText(image, 'Object', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 0, 0), 1)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 1)

# Display the image with detections
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
