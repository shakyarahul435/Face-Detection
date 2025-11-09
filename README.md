# 👁️ Face Detection App using Streamlit and OpenCV

This project is a simple **real-time face detection system** built using **Streamlit** and **OpenCV**.  
It uses the **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`) to detect faces from the webcam feed and display them on the Streamlit web interface.

---

## 📦 Requirements

Before running the project, make sure you have the following installed:

- Python 3.8 or above  
- Streamlit  
- OpenCV  
- NumPy  
- Matplotlib  

You can install all dependencies using the command below:

```bash
pip install streamlit opencv-python numpy matplotlib
```
---
```
📁 Project Structure
project-folder/
│
├── new.py                               # Main Streamlit app file
├── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
└── README.md                            # Project documentation

```
---
▶️ How to Run

- Clone or download this repository, then open the folder in your terminal.
- Make sure your haarcascade_frontalface_default.xml file is in the same directory as new.py.
---
Run the Streamlit app using the command below:
```bash
streamlit run new.py
```
---
The browser will automatically open the Streamlit interface.
Click on "Open Camera" to start real-time face detection.

⚙️ How It Works
- When you click "Open Camera", the webcam feed is activated.
- The script converts each frame to grayscale.
- OpenCV’s Haar Cascade Classifier detects faces within the frame.
- Detected faces are highlighted with green rectangles and displayed in the app.


Build By:
Rahul Shakya
