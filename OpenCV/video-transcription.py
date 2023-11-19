import os
import cv2
import time
import numpy as np

# Function to calculate the percentage of changed pixels between two frames
def percentage_of_changed_pixels(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    changed_pixels = np.sum(threshold) / (threshold.shape[0] * threshold.shape[1])
    return changed_pixels

# Function to capture screenshots from a video file
def capture_screenshots(video_file, output_folder, interval=2, threshold_percent=0.8):
    cap = cv2.VideoCapture(video_file)
    frame_rate = int(cap.get(5))  # Get the frame rate of the video
    count = 0
    previous_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is not None and count % (frame_rate * interval) == 0:
            changed_pixels = percentage_of_changed_pixels(previous_frame, frame)
            if changed_pixels >= threshold_percent:
                timestamp = int(time.time())
                screenshot_path = os.path.join(output_folder, f"screenshot_{timestamp}.png")
                cv2.imwrite(screenshot_path, frame)
                print(f"Saved screenshot: {screenshot_path}")

        previous_frame = frame.copy()
        count += 1

    cap.release()

# Function to iterate over videos in a folder and subfolders
def iterate_over_videos(root_folder, output_folder, interval=2, threshold_percent=0.8):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".mp4"):
                video_file = os.path.join(foldername, filename)
                capture_screenshots(video_file, output_folder, interval, threshold_percent)

if __name__ == "__main__":
    root_folder = "C:\\Users\PC\\Videos\\ChatGPT\\[TutsNode.net] - The 2023 ChatGPT and Prompt Engineering Masterclass"
    output_folder = "D:\\screenshot"
    interval = 2  # Time interval in seconds
    threshold_percent = 0.8  # Minimum percentage of changed pixels

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    iterate_over_videos(root_folder, output_folder, interval, threshold_percent)