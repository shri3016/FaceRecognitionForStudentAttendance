# # For admin: Code for making own dataset of faces detected without facial landmarks and saving them in separate folder 
# # respectively using laptop camera

# import cv2
# import os
# from mtcnn.mtcnn import MTCNN

# # Create folders to store the extracted faces without facial landmarks
# test_images = "C:/Users/natha/Desktop/image_preprocessing/facenetp/data/te_i"

# # prompt the user to input the names
# name = input("Enter the name of the student: ").strip()

# # create a new folder for the person
# person_folder = os.path.join(test_images, name)
# os.makedirs(person_folder, exist_ok=True)

# # create the detector, using default weights
# detector = MTCNN()

# # Dictionary to store the number of times each face has been detected
# face_counts = {}

# # Counter to track the number of extracted faces
# face_counter = 0

# os.makedirs(test_images, exist_ok=True)

# # Open the default camera (ID=0)
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the camera stream
#     ret, img = cap.read()
#     img_copy = img.copy()

#     # Detect faces in the frame
#     faces = detector.detect_faces(img)

#     for i, face in enumerate(faces):
#         x, y, w, h = face['box']

#         # Draw the bounding box
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)
#         cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)

#         # Draw the facial landmarks
#         for key, value in face['keypoints'].items():
#             cv2.circle(img, value, 2, (0, 255, 0), 2)

#         # Crop the image to extract the face with landmarks
#         face_img_with_landmarks = img[y:y+h, x:x+w]
#         face_img_with_landmarks_copy = img_copy[y:y+h, x:x+w]

#         # Save the extracted face without landmarks
#         face_without_landmarks_file_path = os.path.join(test_images, f"face_{face_counter}.png")
#         cv2.imwrite(face_without_landmarks_file_path, face_img_with_landmarks_copy)

#         # Increment the face counter
#         face_counter += 1

#     # create folder to store cropped images
#     cropped_folder_camera = "C:/Users/natha/Desktop/image_preprocessing/facenetp/data/test_images/cropped_images"
#     os.makedirs(cropped_folder_camera, exist_ok=True)

#     # loop through the extracted faces and crop them to 160x160
#     for i in range(face_counter):
#         # read the face image
#         face_file_path = os.path.join(test_images, f"face_{i}.png")
#         face_img = cv2.imread(face_file_path)

#         # resize the face image to 160x160
#         resized_face_img = cv2.resize(face_img, (160, 160))

#         # save the cropped face image to the cropped images folder
#         cropped_file_path = os.path.join(cropped_folder_camera, f"cropped_face_{i}.png")
#         cv2.imwrite(cropped_file_path, resized_face_img)

#     # Show the processed frame with bounding boxes and facial landmarks
#     cv2.imshow("Face Detection", img)

#     # Wait for the 'q' key to exit the loop
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()




import cv2
from mtcnn.mtcnn import MTCNN
import os

# Folder paths
base_folder = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"
te_i_folder = os.path.join(base_folder, "te_i")
tr_i_folder = os.path.join(base_folder, "tr_i")
te_i_cropped_folder = os.path.join(base_folder, "te_i_cropped")
tr_i_cropped_folder = os.path.join(base_folder, "tr_i_cropped")

# Create folders if they don't exist
os.makedirs(te_i_folder, exist_ok=True)
os.makedirs(tr_i_folder, exist_ok=True)
os.makedirs(te_i_cropped_folder, exist_ok=True)
os.makedirs(tr_i_cropped_folder, exist_ok=True)

# Get student's name from user
person_name = input("Enter student's name: ")

# Create sub-folders based on student's name
te_i_person_folder = os.path.join(te_i_folder, person_name)
tr_i_person_folder = os.path.join(tr_i_folder, person_name)
te_i_cropped_person_folder = os.path.join(te_i_cropped_folder, person_name)
tr_i_cropped_person_folder = os.path.join(tr_i_cropped_folder, person_name)

# Create sub-folders if they don't exist
os.makedirs(te_i_person_folder, exist_ok=True)
os.makedirs(tr_i_person_folder, exist_ok=True)
os.makedirs(te_i_cropped_person_folder, exist_ok=True)
os.makedirs(tr_i_cropped_person_folder, exist_ok=True)

# Create the detector, using default weights
detector = MTCNN()

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Check if camera capture is successful
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Counter for face image numbering
te_i_counter = 0
tr_i_counter = 0

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

        # Save faces to te_i folder
        face_copy_pixels = frame[y:y+h, x:x+w]
        face_filename = os.path.join(te_i_person_folder, "face_" + str(te_i_counter) + ".png")
        cv2.imwrite(face_filename, face_copy_pixels)
        te_i_counter += 1

        # Save faces to tr_i folder
        face_copy_pixels = frame[y:y+h, x:x+w]
        face_filename = os.path.join(tr_i_person_folder, "face_" + str(tr_i_counter) + ".png")
        cv2.imwrite(face_filename, face_copy_pixels)
        tr_i_counter += 1

        # Save cropped faces to te_i_cropped folder
        face_copy_pixels = frame[y:y+h, x:x+w]
        face_copy_pixels = cv2.resize(face_copy_pixels, (160, 160))
        face_filename = os.path.join(te_i_cropped_person_folder, "face_" + str(te_i_counter) + ".png")
        cv2.imwrite(face_filename, face_copy_pixels)
        te_i_counter += 1

        # Save cropped faces to tr_i_cropped folder
        face_copy_pixels = frame[y:y+h, x:x+w]
        face_copy_pixels = cv2.resize(face_copy_pixels, (160, 160))
        face_filename = os.path.join(tr_i_cropped_person_folder, "face_" + str(tr_i_counter) + ".png")
        cv2.imwrite(face_filename, face_copy_pixels)
        tr_i_counter += 1

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