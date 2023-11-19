import cv2

# Read the video file
video = cv2.VideoCapture('video.mp4')

# Create a tracker object (you can choose different trackers)
tracker = cv2.TrackerCSRT_create()

# Read the first frame
ret, frame = video.read()

# Select a ROI (Region of Interest)
bbox = cv2.selectROI("Select Object", frame, False)

# Initialize the tracker with the first frame and ROI
tracker.init(frame, bbox)

# Process frames in the video
while True:
    # Read the next frame
    ret, frame = video.read()

    # If video has ended, break the loop
    if not ret:
        break

    # Update the tracker with the current frame
    ret, bbox = tracker.update(frame)

    # Draw the bounding box around the tracked object
    if ret:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Tracking", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video and close windows
video.release()
cv2.destroyAllWindows()
