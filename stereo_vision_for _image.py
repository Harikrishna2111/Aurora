import cv2
import numpy as np

# Load calibration parameters
calib_data = np.load('stereo_calib.npz')
cameraMatrixL = calib_data['cameraMatrixL']
distCoeffsL = calib_data['distCoeffsL']
cameraMatrixR = calib_data['cameraMatrixR']
distCoeffsR = calib_data['distCoeffsR']
R1 = calib_data['R1']
R2 = calib_data['R2']
P1 = calib_data['P1']
P2 = calib_data['P2']
Q = calib_data['Q']

# Load your stereo images
img_left = cv2.imread('path/to/your/left/1.png')
img_right = cv2.imread('path/to/your/right/1.png')

# Rectification maps
h, w = img_left.shape[:2]
mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (w, h), cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (w, h), cv2.CV_32FC1)

# Remap images
rectified_left = cv2.remap(img_left, mapL1, mapL2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, mapR1, mapR2, cv2.INTER_LINEAR)

# Show rectified images
cv2.imshow('Rectified Left', rectified_left)
cv2.imshow('Rectified Right', rectified_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
