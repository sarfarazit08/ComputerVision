import cv2

# Open the video file
video = cv2.VideoCapture('video2.mp4')

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If the frame was not read successfully, exit the loop
    if not ret:
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Wait for the 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the windows
video.release()
cv2.destroyAllWindows()
