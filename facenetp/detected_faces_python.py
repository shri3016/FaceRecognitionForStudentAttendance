import cv2
import os
import shutil
from mtcnn.mtcnn import MTCNN

# Define the folder name for storing the detected faces
detected_faces_folder = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces"

# Check if the detected_faces folder already exists
if os.path.exists(detected_faces_folder):
    # If the folder exists, delete it and its contents
    shutil.rmtree(detected_faces_folder)

# Create a new folder to store the detected faces
os.makedirs(detected_faces_folder)

# Create the detector, using default weights
detector = MTCNN()

# Open the default camera (ID=0)
cap = cv2.VideoCapture("rtsp://192.168.123.211:554/user=admin_password=8M0sJUeP_channel=0_stream=0.sdp?real_stream")

# Counter to track the number of saved faces
face_counter = 0

while True:
    # Read a frame from the camera stream
    ret, img = cap.read()

    # Detect faces in the frame
    faces = detector.detect_faces(img)

    for i, face in enumerate(faces):
        x, y, w, h = face['box']

        # Draw the bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)

        # Crop the image to extract the face
        face_img = img[y:y+h, x:x+w]

        # Save the extracted face with a unique name
        face_file_path = os.path.join(detected_faces_folder, f"{face_counter}.png")
        cv2.imwrite(face_file_path, face_img)

        # Increment the face counter
        face_counter += 1

    # Show the processed frame with bounding boxes
    cv2.imshow("Face Detection", img)

    # Wait for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()