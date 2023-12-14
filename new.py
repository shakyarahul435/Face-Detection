import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def detect_faces(frame):
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade for face detection
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = haar_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def main():
    st.title('Image Recognition System')
    st.write("**Click on the button to open the camera and detect faces.**")

    if st.button("Open Camera"):
        cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not cap.isOpened():
            st.error("Error: Unable to open camera.")
            return

        start_time_temp = time.time()
        duration = 10  

        #while True:
        for i in range(1):
            
                ret, frame = cap.read()

                # Break the loop if there's an issue reading the frame
                if not ret:
                    st.error("Error: Unable to read frame.")
                    break

                # Detect faces and update the frame
                frame_with_faces = detect_faces(frame)

                # Display the updated frame using st.image
                st.image(frame_with_faces, channels="BGR", use_column_width=True, caption="Frame with Detected Faces")
            
        cap.release()

if __name__ == "__main__":
    main()
