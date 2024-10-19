import cv2
import numpy as np

# Load your left and right images
left_img = cv2.imread('undistorted_left.png', 0)
right_img = cv2.imread('undistorted_right.png', 0)

# Create a StereoSGBM object
min_disp = 0
num_disp = 16 * 6  # Needs to be divisible by 16
block_size = 11  # Match the window size for block matching

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=200,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# Load the WLS filter from the ximgproc module
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
disparity_right = cv2.ximgproc.createRightMatcher(stereo).compute(right_img, left_img)
disparity_right = np.int16(disparity_right)

# Apply WLS filter
filtered_disp = wls_filter.filter(disparity, left_img, disparity_map_right=disparity_right)

# Normalize for visualization
filtered_disp = cv2.normalize(filtered_disp, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
filtered_disp = np.uint8(filtered_disp)

# Save or display the result
cv2.imshow('Filtered Disparity Map', filtered_disp)
cv2.imwrite('filtered_disparity_map.jpg', filtered_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
