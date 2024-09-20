# This script splits a video into two halves: top and bottom. 
# It reads a video file, splits each frame into two halves, and writes the top and bottom halves to two separate video files. 
# The output videos are saved as 'top_half_video.mp4' and 'bottom_half_video.mp4'.

import cv2

# Load the video
cap = cv2.VideoCapture("path_to_video.mp4")

# Get the width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region of interest (ROI) for the top and bottom halves
top_roi = (0, 0, width, height // 2)
bottom_roi = (0, height // 2, width, height)

# Create VideoWriters for the output videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
top_out = cv2.VideoWriter('top_half_video.mp4', fourcc, fps, (width, height // 2))
bottom_out = cv2.VideoWriter('bottom_half_video.mp4', fourcc, fps, (width, height // 2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame into top and bottom halves
    top_frame = frame[top_roi[1]:top_roi[3], top_roi[0]:top_roi[2]]
    bottom_frame = frame[bottom_roi[1]:bottom_roi[3], bottom_roi[0]:bottom_roi[2]]

    # Write the frames to the output videos
    top_out.write(top_frame)
    bottom_out.write(bottom_frame)

# Release the resources
cap.release()
top_out.release()
bottom_out.release()
cv2.destroyAllWindows()
