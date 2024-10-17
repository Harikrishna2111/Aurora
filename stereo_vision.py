import cv2
import numpy as np

# Load the saved stereo calibration data
calib_data = np.load('stereo_calib.npz')
cameraMatrixL = calib_data['cameraMatrixL']
distCoeffsL = calib_data['distCoeffsL']
cameraMatrixR = calib_data['cameraMatrixR']
distCoeffsR = calib_data['distCoeffsR']
R = calib_data['R']
T = calib_data['T']
R1 = calib_data['R1']
R2 = calib_data['R2']
P1 = calib_data['P1']
P2 = calib_data['P2']
Q = calib_data['Q']

# Read stereo images (left and right)
imgL = cv2.imread('path_to_left_image.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with actual path
imgR = cv2.imread('path_to_right_image.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with actual path

# Rectify the images
h, w = imgL.shape[:2]
mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (w, h), cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (w, h), cv2.CV_16SC2)

rectifiedL = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

# Show rectified images for verification
cv2.imshow('Rectified Left', rectifiedL)
cv2.imshow('Rectified Right', rectifiedR)
cv2.waitKey(0)
cv2.destroyAllWindows()

# StereoSGBM Parameters
min_disparity = 0
num_disparities = 16 * 5  # Should be divisible by 16
block_size = 5  # Must be an odd number (e.g., 5, 7, 9)

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=50,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map
disparity_map = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

# Display the disparity map
cv2.imshow('Disparity Map', (disparity_map - min_disparity) / num_disparities)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reproject disparity to 3D space using the Q matrix
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

# Mask out points where disparity is less than or equal to 0
mask_map = disparity_map > disparity_map.min()

# Extract the valid 3D points
output_points = points_3D[mask_map]

# Show the depth map (optional, based on disparity)
depth_map = Q[2, 3] / (disparity_map + 1e-6)  # Avoid division by zero
cv2.imshow('Depth Map', depth_map / np.max(depth_map))
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Depth estimation completed.")
