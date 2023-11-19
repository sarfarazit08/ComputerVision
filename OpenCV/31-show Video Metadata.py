import cv2

video = cv2.VideoCapture('video2.mp4')

# Get the frames per second (FPS) of the video
fps = video.get(cv2.CAP_PROP_FPS)
print('FPS:', fps)

# Get the resolution of the video
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames in the video.
FourCC_code_videoCodec = int(video.get(cv2.CAP_PROP_FOURCC)) # FourCC code representing the video codec.
video_duration_in_seconds = int(total_number_of_frames / fps) # Duration of the video in seconds.
video_stream_format = int(video.get(cv2.CAP_PROP_FORMAT)) # Format of the video stream.
video_brightness_level = int(video.get(cv2.CAP_PROP_BRIGHTNESS)) # Brightness level of the video.
video_contrast_level = int(video.get(cv2.CAP_PROP_CONTRAST)) # Contrast level of the video.
video_saturation_level = int(video.get(cv2.CAP_PROP_SATURATION)) # Saturation level of the video.
video_hue_level = int(video.get(cv2.CAP_PROP_HUE)) # Hue level of the video.
video_exposure_level  = int(video.get(cv2.CAP_PROP_EXPOSURE)) # Exposure level of the video


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Add metadata to the video frame
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, 'Resolution: {}x{}'.format(width, height), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f'total_number_of_frames: {total_number_of_frames}', (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f'FourCC_code_videoCodec: {FourCC_code_videoCodec}', (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f'video_duration_in_seconds: {video_duration_in_seconds}', (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    # cv2.putText(frame, f'video_stream_format: {video_stream_format}', (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    # cv2.putText(frame, f'video_brightness_level: {video_brightness_level}', (10, 105), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    # cv2.putText(frame, f'video_contrast_level: {video_contrast_level}', (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    # cv2.putText(frame, f'video_saturation_level: {video_saturation_level}', (10, 135), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    # cv2.putText(frame, f'video_hue_level: {video_hue_level}', (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
    # cv2.putText(frame, f'video_exposure_level: {video_exposure_level}', (10, 165), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)


    # Display the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
