import cv2

# RTSP URLs for both camera channels
url1 = 'rtsp://admin:aurora123@192.168.1.200:554/cam/realmonitor?channel=1&subtype=0'
url2 = 'rtsp://admin:aurora123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0'

# Create VideoCapture objects for both cameras
cap1 = cv2.VideoCapture(url1)
cap2 = cv2.VideoCapture(url2)

# Check if the first camera opened successfully
if not cap1.isOpened():
    print("Error: Could not open the first video stream")
    exit()

# Check if the second camera opened successfully
if not cap2.isOpened():
    print("Error: Could not open the second video stream")
    exit()

# Define the window name
window_name = 'Camera Feeds Side by Side'

while True:
    # Capture frame-by-frame from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Check if frames were grabbed successfully
    if not ret1 or not ret2:
        print("Error: Failed to grab frames from one of the cameras")
        break

    # Resize the frames (optional: adjust dimensions based on your screen size)
    frame1_resized = cv2.resize(frame1, (640, 480))
    frame2_resized = cv2.resize(frame2, (640, 480))

    # Concatenate the frames horizontally
    combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

    # Display the combined frame
    cv2.imshow(window_name, combined_frame)

    # Exit the video window on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captures and close the windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
