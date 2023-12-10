import streamlit as st
import cv2
import numpy as np



def main():
    st.title('Image recognization System ')
    st.write("**Click on the button to open the camera and detect image.**")


    if st.button("Open Camera"):
        cap = cv2.VideoCapture(0)
        for i in range(1):
            
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            st.image(frame_gray)

            haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = haar_cascade.detectMultiScale(frame_gray, 1.3, 5)
            
            st.write("This is rectangle face",faces)
            for(x,y,w,h) in faces:
                st.write("This is a length of rectange",x)
                cv2.rectangle(frame_gray, (x,y), (x+w,y+h), (0,255,0),2)
                st.write("THis is working")
                st.image(frame, channels="BGR")
                

        cap.release()
        st.write("This is a image of man.")
       

if __name__ == "__main__":
    main()