import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture("video2.mp4") # webcame video capture

fig, axes = plt.subplots(nrows=2, ncols=4)

while video.isOpened():
    ret, frame = video.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    xyz = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
    luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Plot on axes
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB')

    axes[0, 1].imshow(gray)
    axes[0, 1].set_title('gray')

    axes[0, 2].imshow(hsl)
    axes[0, 2].set_title('hsl')

    axes[0, 3].imshow(lab)
    axes[0, 3].set_title('lab')

    axes[1, 3].imshow(yuv)
    axes[1, 3].set_title('yuv')

    axes[1, 1].imshow(xyz)
    axes[1, 1].set_title('xyz')

    axes[1, 2].imshow(luv)
    axes[1, 2].set_title('luv')

    axes[1, 0].imshow(ycrcb)
    axes[1, 0].set_title('ycrcb')

    plt.pause(0.1)
    plt.draw()
plt.close()
video.release()
cv2.destroyAllWindows()

