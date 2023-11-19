import cv2

video = cv2.VideoCapture(0) # webcame video capture

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Apply any desired image processing operations on the frame
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ) # or any number ranging from 1 to 255
    '''
        cv.COLOR_BGR2GRAY: Converts BGR to grayscale color space.
        cv2.COLOR_BGR2RGB: Converts BGR to RGB color space.
        cv2.COLOR_BGR2HSV: Converts BGR to HSV (Hue-Saturation-Value) color space.
        cv2.COLOR_BGR2HLS: Converts BGR to HLS (Hue-Lightness-Saturation) color space.
        cv2.COLOR_BGR2Lab: Converts BGR to CIE Lab color space.
        cv2.COLOR_BGR2LUV: Converts BGR to CIE Luv color space.
        cv2.COLOR_BGR2YUV: Converts BGR to YUV color space.
        cv2.COLOR_BGR2XYZ: Converts BGR to CIE XYZ color space.
        cv2.COLOR_BGR2YCrCb: Converts BGR to YCrCb color space.
    '''
    cv2.imshow('Video', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
