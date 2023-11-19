import cv2
import numpy as np

video = cv2.VideoCapture('video.mp4')

# Read the first frame
_, frame = video.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Create an empty mask for optical flow visualization
mask = np.zeros_like(frame)

while True:
    # Read the current frame
    _, frame = video.read()
    if frame is None:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualize the optical flow
    mask = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask[..., 2] = 0
    mask[..., 0] += flow[..., 0]
    mask[..., 1] += flow[..., 1]

    # Display the frame with optical flow
    cv2.imshow('Optical Flow', mask)
    if cv2.waitKey(1) == ord('q'):
        break

    # Update the previous frame
    prev_gray = gray

video.release()
cv2.destroyAllWindows()
