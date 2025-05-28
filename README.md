# 👁️‍🗨️ Face Detection using OpenCV

This project implements a simple face detection system using Python and OpenCV. It provides two modes:

1. **Detect faces in a static image**
2. **Detect faces in real-time using your webcam**

## 📌 Features

* Uses **Haar Cascade Classifier** for face detection.
* Detects multiple faces in an image or webcam feed.
* Highlights detected faces with bounding boxes.
* Displays the number of faces detected in webcam mode.
* Easy-to-use CLI interface.

## 🛠️ Technologies Used

* Python
* OpenCV (`cv2`)
* NumPy

## 🚀 How It Works

### 1. Static Image Detection

* Converts the image to grayscale.
* Loads a pre-trained Haar Cascade model.
* Detects faces using `detectMultiScale`.
* Draws bounding boxes around detected faces.
* Displays the final image with detected faces.

### 2. Real-Time Webcam Detection

* Captures video stream from the default webcam.
* Continuously converts each frame to grayscale.
* Detects faces and draws bounding boxes in real-time.
* Displays the number of detected faces on the video feed.
* Exits when the user presses the `q` key.

## 📦 Setup Instructions

1. **Clone the repository or copy the code.**
2. **Install dependencies:**

   ```bash
   pip install opencv-python numpy
   ```
3. **Run the script:**

   ```bash
   python face_detection.py
   ```

## 🖼️ Example Usage

### ➤ Detecting Faces in an Image

```bash
python face_detection.py
# Select option 1
# Enter the path to your image file when prompted
```

### ➤ Detecting Faces via Webcam

```bash
python face_detection.py
# Select option 2 to start webcam-based face detection
# Press 'q' to stop
```

## 📁 Haar Cascade Classifier

This project uses OpenCV’s built-in Haar Cascade file:
`haarcascade_frontalface_default.xml`
It is loaded from OpenCV’s data module:

```python
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
```

## ✅ Output

* Static Image: Opens a window showing faces highlighted with blue rectangles.
* Webcam: Opens a live feed showing faces in green rectangles and counts them.


