import cv2


# Define the size of the chessboard pattern (number of inner corners per row and column)
chessboard_size = (7, 7)  # Adjust to match your actual chessboard size
def find_corners(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

    # If corners are found, draw them on the image
    if ret:
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)


    # Output the corners array if detected
    return ret, corners, gray_image
