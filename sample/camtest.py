import numpy as np
import cv2

# Check for left and right camera IDs (these values can change)
CamL_id = 'rtsp://admin:aurora123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0'  # Camera ID for left camera
CamR_id = 'rtsp://admin:aurora123@192.168.1.200:554/cam/realmonitor?channel=1&subtype=0'  # Camera ID for right camera

# Open both cameras
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Read the stereo rectification maps from the XML file
cv_file = cv2.FileStorage("data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

# Create a window with trackbars for tuning the stereo parameters
cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

# Create trackbars for tuning stereo matching parameters
cv2.createTrackbar('numDisparities', 'disp', 1, 17, lambda x: None)  # Max disparity range
cv2.createTrackbar('blockSize', 'disp', 5, 50, lambda x: None)  # Block size
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, lambda x: None)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, lambda x: None)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, lambda x: None)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, lambda x: None)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, lambda x: None)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, lambda x: None)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, lambda x: None)

# Create StereoBM object for disparity calculation
stereo = cv2.StereoBM_create()

while True:
    # Capture frames from both cameras
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    if retL and retR:
        # Convert both frames to grayscale
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Apply stereo rectification to both images
        Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Read stereo parameters from the trackbars
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Set the parameters for stereo computation
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Compute the disparity map
        disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0

        # Normalize the disparity map for better visualization
        disparity = (disparity - minDisparity) / numDisparities
        cv2.imshow('disp', disparity)

        # Break the loop on pressing 'Esc'
        if cv2.waitKey(1) == 27:
            break

# Release the cameras and close windows
CamL.release()
CamR.release()
cv2.destroyAllWindows()
