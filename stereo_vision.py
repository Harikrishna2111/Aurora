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

# Open video streams for the two cameras (camera 0 and camera 1)
capL = cv2.VideoCapture(0)  # Left camera
capR = cv2.VideoCapture(1)  # Right camera

# Set resolution for both cameras (optional)
frame_width = 640
frame_height = 480
capL.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# StereoSGBM Parameters
min_disparity = 0
num_disparities = 16 * 5  # Must be divisible by 16
block_size = 5  # Odd number

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

while capL.isOpened() and capR.isOpened():
    retL, frameL = capL.read()  # Capture frame from the left camera
    retR, frameR = capR.read()  # Capture frame from the right camera

    if not retL or not retR:
        print("Error capturing frames from cameras")
        break

    # Convert frames to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Get frame dimensions
    h, w = grayL.shape[:2]

    # Rectify the images
    mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (w, h), cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (w, h), cv2.CV_16SC2)

    rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Compute disparity map
    disparity_map = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

    # Normalize disparity map for display
    disp_display = (disparity_map - min_disparity) / num_disparities
    disp_display = cv2.normalize(disp_display, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # Display rectified images and disparity map
    cv2.imshow('Rectified Left', rectifiedL)
    cv2.imshow('Rectified Right', rectifiedR)
    cv2.imshow('Disparity Map', disp_display)

    # Reproject disparity to 3D space using the Q matrix
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    # Mask out points where disparity is less than or equal to 0
    mask_map = disparity_map > disparity_map.min()

    # Extract valid 3D points
    output_points = points_3D[mask_map]

    # Optional: Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close windows
capL.release()
capR.release()
cv2.destroyAllWindows()
