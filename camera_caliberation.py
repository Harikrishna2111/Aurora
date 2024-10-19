import cv2
import numpy as np
import glob
from corner_finder import find_corners

# Chessboard dimensions
chessboard_size = (7, 7)
square_size = 2  # Change this to the size of a square on your chessboard in real-world units

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints_left = []  # 2D points in image plane (left camera)
imgpoints_right = []  # 2D points in image plane (right camera)

# Load stereo images
images_left = glob.glob('D:/Projects/Aurora/left/*.png')  # Change to your image path
images_right = glob.glob('D:/Projects/Aurora/right/*.png')  # Change to your image path

for img_left, img_right in zip(images_left, images_right):
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)

    retL, cornersL, grayL= find_corners(imgL)
    retR, cornersR, grayR= find_corners(imgR)


    if retL and retR:
        objpoints.append(objp)

        # Refine corner locations
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        # Visualize corners
        cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
        cv2.imshow('Left Camera', imgL)
        cv2.imshow('Right Camera', imgR)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate individual cameras
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# Stereo calibration (find extrinsic parameters)
retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)

# Stereo rectification
rectify_scale = 1  # 0: crop, 1: keep full size
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, grayL.shape[::-1], R, T, alpha=rectify_scale)

# Save the calibration results
np.savez('stereo_calib.npz', cameraMatrixL=cameraMatrixL, distCoeffsL=distCoeffsL,
         cameraMatrixR=cameraMatrixR, distCoeffsR=distCoeffsR, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("Stereo calibration completed.")
