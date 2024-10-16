import cv2
import numpy as np

# Chessboard dimensions
chessboard_size = (9, 6)

# Termination criteria for refining the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points for both cameras
objpoints = []  # 3D points in real-world space
imgpoints_left = []  # 2D points in image plane of left camera
imgpoints_right = []  # 2D points in image plane of right camera

# Initialize camera captures (replace with your camera IDs)
CamL_id = 'rtsp://admin:aurora123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0'
CamR_id = 'rtsp://admin:aurora123@192.168.1.200:554/cam/realmonitor?channel=1&subtype=0'
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

while True:
    # Capture images from both cameras
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    if not retL or not retR:
        print("Error: Could not capture images")
        break

    # Convert to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners for both images
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        # Refine the corner positions
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)

        cv2.imshow('Left Camera', imgL)
        cv2.imshow('Right Camera', imgR)

        # Wait for user to press 's' to save the points
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Saving chessboard data for stereo calibration...")
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

        # Exit when 'q' is pressed
        elif key == ord('q'):
            break

# Calibrate the individual cameras
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria, flags=flags
)

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T
)

# Generate rectification maps for both cameras
Left_Stereo_Map_x, Left_Stereo_Map_y = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map_x, Right_Stereo_Map_y = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

# Save the rectification maps to an XML file
cv_file = cv2.FileStorage("data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map_x)
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map_y)
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map_x)
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map_y)
cv_file.release()

print("Stereo calibration and rectification complete. Maps saved to 'stereo_rectify_maps.xml'.")

# Release resources
CamL.release()
CamR.release()
cv2.destroyAllWindows()
