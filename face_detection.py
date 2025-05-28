import cv2
import numpy as np

def detect_faces_in_image(image_path):
    """
    Detect faces in a static image
    """
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Convert to grayscale (face detection works better on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Found {len(faces)} face(s)")
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the result
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()

def detect_faces_webcam():
    """
    Detect faces using webcam in real-time
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add text showing number of faces
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Detection - Webcam', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Face Detection Project")
    print("1. Detect faces in image")
    print("2. Detect faces using webcam")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        image_path = input("Enter path to image file: ")
        detect_faces_in_image(image_path)
    elif choice == "2":
        detect_faces_webcam()
    else:
        print("Invalid choice!")