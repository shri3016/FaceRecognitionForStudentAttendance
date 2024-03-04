import os

import cv2
from mtcnn.mtcnn import MTCNN

# Folder paths
base_folder = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"
train_images_folder = os.path.join(base_folder, "train_images")
train_images_cropped_folder = os.path.join(base_folder, "train_images_cropped")

# Create folders if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_images_cropped_folder, exist_ok=True)

# Get student's name from user
person_name = input("Enter student's name: ")

# Check if student's name folder already exists
train_images_student_folder = os.path.join(train_images_folder, person_name)
if os.path.exists(train_images_student_folder):
    print("The student's name already exists.")
    exit()

# Create sub-folders based on student's name
train_images_student_folder = os.path.join(train_images_folder, person_name)
train_images_cropped_person_folder = os.path.join(train_images_cropped_folder, person_name)

# Create sub-folders if they don't exist
os.makedirs(train_images_student_folder, exist_ok=True)
os.makedirs(train_images_cropped_person_folder, exist_ok=True)

# Create the detector, using default weights
detector = MTCNN()

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Check if camera capture is successful
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Counter for face image numbering
train_images_counter = 0

# Capture frames from the camera feed
while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Break loop if frame reading is unsuccessful
    if not ret:
        print("Failed to read frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Process detected faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Save faces to tr_i folder
        face_copy_pixels = frame[y:y+h, x:x+w]
        face_filename = os.path.join(train_images_student_folder, str(train_images_counter) + ".png")
        cv2.imwrite(face_filename, face_copy_pixels)
        train_images_counter += 1

        # Save cropped faces to tr_i_cropped folder
        face_copy_pixels = frame[y:y+h, x:x+w]
        face_copy_pixels = cv2.resize(face_copy_pixels, (160, 160))
        face_filename = os.path.join(train_images_cropped_person_folder, str(train_images_counter) + ".png")
        cv2.imwrite(face_filename, face_copy_pixels)
        train_images_counter += 1

        # Draw facial landmarks
        for key, value in face['keypoints'].items():
            cv2.circle(frame, value, 2, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Landmarks', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture and close windows
cap.release()
cv2.destroyAllWindows()