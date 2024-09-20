import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model # type: ignore
import tempfile
import os

# Function to load YOLO model
def load_yolo_model(model_path):
    path = f"object_detection_model/best_{model_path}.pt"
    model = YOLO(path)
    model.to('cuda')
    return model

# Function to load CNN model
def load_cnn_model(model_path):
    path = f"classification_model/best_{model_path}.h5"
    model = load_model(path)
    return model

# Streamlit app
st.title("Object Detection and Classification")

# Dropdown menu for selecting YOLO or classification model
yolonclassification_model_option = st.selectbox(
    "Select YOLO Model",
    ("capsule", "candle", "cashew", "fryums")
)


# Load the selected YOLO and CNN models
yolo_model = load_yolo_model(yolonclassification_model_option)
cnn_model = load_cnn_model(yolonclassification_model_option)

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Load the video file
    video_path = tfile.name
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter to save the output video
    output_video_path = './output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    # Process video frames
    while video.isOpened():
        ret, frame = video.read()
        frame_count += 1
        
        # Break the loop if any video frame is not read properly
        if not ret:
            break
        
        # Skip frames to increase processing speed
        if frame_count % frame_skip != 0:
            continue
        
        # Reduce input resolution for faster processing
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Detect objects in the frame using YOLO model
        results = yolo_model(frame)
        
        # Iterate through detected objects and classify them
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the detected object from the frame
                cropped_frame = frame[y1:y2, x1:x2]
                
                # Preprocess the cropped frame for CNN model
                cropped_frame_resized = cv2.resize(cropped_frame, (224, 224))  # Assuming CNN input size is 224x224
                cropped_frame_resized = cropped_frame_resized.astype('float32') / 255.0
                cropped_frame_resized = np.expand_dims(cropped_frame_resized, axis=0)
                
                # Classify the cropped frame using CNN model
                prediction = cnn_model.predict(cropped_frame_resized)
                class_label = np.argmax(prediction, axis=1)[0]
                
                # Determine the color of the bounding box
                if class_label == 0:  # Assuming 0 is the label for defective
                    contour_color = (0, 0, 255)  # Red for defective
                else:
                    contour_color = (0, 255, 0)  # Green for non-defective
                
                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), contour_color, 2)
        
        # Write the frame with bounding boxes to the output video
        out.write(frame)
        
        # Display the frame with bounding boxes (optional)
        st.image(frame, channels="BGR")
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects and close all OpenCV windows
    video.release()
    out.release()
    cv2.destroyAllWindows()

    # Display the processed video
    st.video(output_video_path)