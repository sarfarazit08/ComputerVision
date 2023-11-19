import numpy as np
import cv2

# Set the number of corners in the calibration pattern
pattern_size = (9, 6)

# Define the square size of the calibration pattern (in any unit)
square_size = 25.0

# Provide the paths to your calibration images
calibration_image_paths = [
    'faces\group.jpg',
    'faces\image.jpg',
    'faces\image2.jpg',
    # Add more image paths as needed
]

# Prepare object points with the known dimensions of the calibration pattern
object_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all calibration images
object_points_list = []  # 3D points in the calibration pattern coordinate space
image_points_list = []  # 2D points in the image plane

# Load calibration images and detect the calibration pattern
for img_path in calibration_image_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Add object points and image points to the lists
        object_points_list.append(object_points)
        image_points_list.append(corners)

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points_list, image_points_list, gray.shape[::-1], None, None
)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
