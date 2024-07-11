import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import tempfile
print(cv2.__version__)
# Load the trained model
model_path = 'D:\Activity Recognition\Activity-Recongnition-Web-Application\CNN_LSTM.h5'
activity_model = load_model(model_path)
 
# Define activity labels
activity_labels = ['WalkingWithDog', 'TaiChi', 'Swing', 'HorseRace']
# Function to preprocess image for CNN input
def preprocess_image(img):
    # Resize image if needed
    img = cv2.resize(img, (64, 64))
    # Convert image to array and preprocess for MobileNetV2
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict activity from a sequence of frames (video)
def predict_activity(frames):
    # Ensure the sequence has exactly 20 frames
    required_length = 20
    if len(frames) > required_length:
        frames = frames[:required_length]
    elif len(frames) < required_length:
        frames += [np.zeros_like(frames[0]) for _ in range(required_length - len(frames))]

    # Preprocess each frame
    preprocessed_frames = [preprocess_image(frame) for frame in frames]
    # Stack frames to form a sequence (time dimension)
    sequence_frames = np.stack(preprocessed_frames)
    sequence_frames = np.expand_dims(sequence_frames, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = activity_model.predict(sequence_frames)
    predicted_label = activity_labels[np.argmax(prediction)]
    
    return predicted_label

# Streamlit app layout
st.title('Activity Recognition Application')

# File uploader for input video
uploaded_file = st.file_uploader("Select A Video File", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    # OpenCV video capture from temporary file
    cap = cv2.VideoCapture(tmp_file_path)
    frames = []
    
    # Read and process frames from video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    # Close video capture
    cap.release()
    
    # Display first frame of the video
    if frames:
        st.image(frames[0], caption='Uploaded Video Frame', use_column_width=True)
    
        # Make prediction
        label = predict_activity(frames)
        st.write(f"Predicted Activity: {label}")
    else:
        st.write("Could not process the video.")
