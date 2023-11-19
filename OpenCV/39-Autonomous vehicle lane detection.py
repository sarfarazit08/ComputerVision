import cv2
import numpy as np

# Load the video for lane detection
video = cv2.VideoCapture('car.mp4')

# Define parameter values
threshold1 = 50
threshold2 = 150
rho = 1
theta = np.pi/180
threshold = 50
min_line_length = 100
max_line_gap = 50

while True:
    # Read a frame from the video
    ret, frame = video.read()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # Perform region of interest selection
    height, width = edges.shape[:2]
    mask = np.zeros_like(edges)
    region_of_interest = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, region_of_interest, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Perform Hough line detection
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Draw the detected lane lines onto the frame
    line_image = np.zeros_like(frame)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Combine the lane lines with the original frame
    lane_image = cv2.addWeighted(frame, 1, line_image, 1, 0)

    # Display the output frame
    cv2.imshow('Lane Detection', lane_image)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
video.release()
cv2.destroyAllWindows()
