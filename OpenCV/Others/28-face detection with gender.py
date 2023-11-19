import cv2

# Load the pre-trained models for face detection and gender classification
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_classifier = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Load the image
image = cv2.imread('image2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over the detected faces
for (x, y, w, h) in faces:
    # Extract the face ROI (Region of Interest)
    face_roi = image[y:y+h, x:x+w]

    # Preprocess the face ROI for gender classification
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Feed the face ROI through the gender classifier
    gender_classifier.setInput(blob)
    gender_predictions = gender_classifier.forward()

    # Get the gender label with the highest confidence
    gender_label = "Male" if gender_predictions[0][0] > gender_predictions[0][1] else "Female"

    # Draw bounding box and gender label on the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0) if gender_label == "Male" else (0, 0, 255), 1)
    cv2.putText(image, gender_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0) if gender_label == "Male" else (0, 0, 255), 1)

# Display the image with face detections and gender labels
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
