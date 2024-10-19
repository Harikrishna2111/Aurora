import cv2
import numpy as np

# Chessboard dimensions (9x6 or other as required)
chessboard_size = (9, 6)

# Termination criteria for refining the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from both cameras
objpoints = []  # 3D points in real world space
imgpointsL = []  # 2D points in image plane of left camera
imgpointsR = []  # 2D points in image plane of right camera

# Initialize left and right camera captures
CamL = cv2.VideoCapture('rtsp://admin:aurora123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0')
CamR = cv2.VideoCapture('rtsp://admin:aurora123@192.168.1.200:554/cam/realmonitor?channel=1&subtype=0')

while True:
    # Capture images from both cameras
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    if not retL or not retR:
        print("Error capturing images from cameras.")
        break

    # Convert images to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    # If both corners are found, refine them
    if retL and retR:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)

        # Show combined camera views
        combined_view = np.hstack((imgL, imgR))
        cv2.imshow('Stereo Calibration', combined_view)

        # Wait for key press 's' to save the points, 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Saving calibration data for both cameras.")
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
        elif key == ord('q'):
            break

# Calibration for both cameras
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Stereo calibration
ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
)

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T)

# Generate rectification maps
Left_Stereo_Map_x, Left_Stereo_Map_y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map_x, Right_Stereo_Map_y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

# Save rectification maps to XML file
cv_file = cv2.FileStorage("data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map_x)
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map_y)
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map_x)
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map_y)
cv_file.release()

print("Stereo calibration complete, maps saved to 'data/stereo_rectify_maps.xml'.")

# Release camera resources
CamL.release()
CamR.release()
cv2.destroyAllWindows()
