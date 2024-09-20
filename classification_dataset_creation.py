# Description: This script reads the top and bottom videos of the capsule dataset and detects the capsules in the top video frames. 
# It then checks if the center of any white spot in the bottom video frames is inside the bounding box of the detected capsule. 
# If so, the bounding box is colored red; otherwise, it is colored green. 
# The frames are saved in the 'capsule_classify' folder with the corresponding color label.

import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLO model with GPU acceleration
model = YOLO('path_to_best_model.pt')
#model.to('cuda') #only if you have GPU and cuda installed

# Load the video files
top_video_path = 'path_to_top_video.mp4'
bottom_video_path = 'path_to_bottom_video.mp4'
top = cv2.VideoCapture(top_video_path)
bottom = cv2.VideoCapture(bottom_video_path)

# Set buffer size for faster video reading
top.set(cv2.CAP_PROP_BUFFERSIZE, 3)
bottom.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# Initialize frame skip, frame count, and image count
frame_skip = 2  # Process every 2nd frame
frame_count = 0
image_count = 0

# Create directories to save classified frames
os.makedirs('./object_classify/0', exist_ok=True)
os.makedirs('./object_classify/1', exist_ok=True)
red = './object_classify/0'
green = './onject_classify/1'

# Process video frames
while top.isOpened() and bottom.isOpened():
    ret1, frame1 = top.read()
    ret2, frame2 = bottom.read()
    frame_count += 1
    
    # Break the loop if any video frame is not read properly
    if not ret1 or not ret2:
        break
    
    # Skip frames to increase processing speed
    if frame_count % frame_skip != 0:
        continue
    
    # Reduce input resolution for faster processing
    frame1 = cv2.resize(frame1, (frame1.shape[1] // 2, frame1.shape[0] // 2))
    frame2 = cv2.resize(frame2, (frame2.shape[1] // 2, frame2.shape[0] // 2))
    
    # Detect objects in the top video frame using YOLO model
    results = model(frame1)
    
    # Convert the bottom frame to grayscale
    gray_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to binary (white spots will be 255, black will be 0)
    _, thresh = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the white spots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours to find the center of each white spot
    white_spot_centers = []
    for contour in contours:
        # Get the moments to calculate the center of the spot
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            white_spot_centers.append((cX, cY))
    
    # Iterate through detected objects and draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            contour_color = (0, 255, 0)  # Default color is green

            # Check if any white spot center is inside the bounding box
            for (cX, cY) in white_spot_centers:
                if x1 <= cX <= x2 and y1 <= cY <= y2:
                    contour_color = (0, 0, 255)  # Change color to red
                    break

            # Draw the bounding box on the frame
            cv2.rectangle(frame1, (x1, y1), (x2, y2), contour_color, 2)

            # Crop the frame within the bounding box
            cropped_frame = frame1[y1+2:y2-2, x1+2:x2-2]
            # Save the cropped frame in the corresponding folder
            if contour_color == (0, 0, 255):
                folder = red
            else:
                folder = green
            cv2.imwrite(os.path.join(folder, f'frame_{image_count}.jpg'), cropped_frame)
            image_count += 1
    
    # Concatenate the top and bottom frames horizontally
    combined_frame = cv2.hconcat([frame1, frame2])
    
    # Display the combined frame (optional)
    cv2.imshow('Video with Bounding Boxes and White Spots', frame1)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close all OpenCV windows
top.release()
bottom.release()
cv2.destroyAllWindows()