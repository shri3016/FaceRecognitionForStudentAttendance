import os #file lib
import cv2 #opencv
from mtcnn.mtcnn import MTCNN 
from flask import Flask, request #
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


# Folder paths
base_folder = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"
train_images_folder = os.path.join(base_folder, "train_images")
train_images_cropped_folder = os.path.join(base_folder, "train_images_cropped")
test_images_folder = os.path.join(base_folder, "test_images")
test_images_cropped_folder = os.path.join(base_folder, "test_images_cropped")

# Create folders if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_images_cropped_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)
os.makedirs(test_images_cropped_folder, exist_ok=True)

def process_first_script(name):
    # Check if student's name folder already exists
    train_images_student_folder = os.path.join(train_images_folder, name)
    if os.path.exists(train_images_student_folder):
        return "The student's name already exists."

    # Create sub-folders based on student's name
    train_images_student_folder = os.path.join(train_images_folder, name)
    train_images_cropped_person_folder = os.path.join(train_images_cropped_folder, name)

    # Create sub-folders if they don't exist
    os.makedirs(train_images_student_folder, exist_ok=True)
    os.makedirs(train_images_cropped_person_folder, exist_ok=True)

    # Capture images using the camera
    cap = cv2.VideoCapture("rtsp://192.168.137.207:554/user=admin_password=IcRKgw5I_channel=0_stream=0.sdp?real_stream")

    # Initialize the MTCNN detector
    detector = MTCNN()

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

            # Save faces to train_images_folder
            face_copy_pixels = frame[y:y+h, x:x+w]
            face_filename = os.path.join(train_images_student_folder, str(train_images_counter) + ".png")
            cv2.imwrite(face_filename, face_copy_pixels)
            train_images_counter += 1

            # Save cropped faces to train_images_cropped_folder
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
    result1 = {
        "message": "First script executed successfully"
    }
    return json.dumps(result1)

def process_second_script(name):
    # Check if student's name folder already exists
    test_images_student_folder = os.path.join(test_images_folder, name)
    if os.path.exists(test_images_student_folder):
        return "The student's name already exists."

    # Create sub-folders based on student's name
    test_images_student_folder = os.path.join(test_images_folder, name)
    test_images_cropped_person_folder = os.path.join(test_images_cropped_folder, name)

    # Create sub-folders if they don't exist
    os.makedirs(test_images_student_folder, exist_ok=True)
    os.makedirs(test_images_cropped_person_folder, exist_ok=True)

    # Initialize camera capture
    cap = cv2.VideoCapture("rtsp://192.168.137.207:554/user=admin_password=IcRKgw5I_channel=0_stream=0.sdp?real_stream")

    # Check if camera capture is successful
    if not cap.isOpened():
        print("Failed to open camera.")
        return "Failed to open camera."

    # Create the detector, using default weights
    detector = MTCNN()

    # Counter for face image numbering
    test_images_counter = 0

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

            # Save faces to test_images_folder
            face_copy_pixels = frame[y:y+h, x:x+w]
            face_filename = os.path.join(test_images_student_folder, str(test_images_counter) + ".png")
            cv2.imwrite(face_filename, face_copy_pixels)
            test_images_counter += 1

            # Save cropped faces to test_images_cropped_folder
            face_copy_pixels = frame[y:y+h, x:x+w]
            face_copy_pixels = cv2.resize(face_copy_pixels, (160, 160))
            face_filename = os.path.join(test_images_cropped_person_folder, str(test_images_counter) + ".png")
            cv2.imwrite(face_filename, face_copy_pixels)
            test_images_counter += 1

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
    result2 = {
        "message": "Second script executed successfully"
    }
    return json.dumps(result2)

    return "Second script executed successfully"

@app.route('/admin-train-images', methods=['POST'])
def admin_train_images():
    name = request.form.get('name')
    result = process_first_script(name)
    return result

@app.route('/admin-test-images', methods=['POST'])
def admin_test_images():
    name = request.form.get('name')
    result = process_second_script(name)
    return result

if __name__ == '__main__':
    app.run(debug=True)
