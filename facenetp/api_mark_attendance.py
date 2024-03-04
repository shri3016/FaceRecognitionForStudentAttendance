import datetime
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request

app = Flask(__name__)

detected_faces_dir = 'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces' 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Recognize faces
def recognize_faces(detected_faces_dir, model, clf, transform, threshold=0.5):
    recognized_faces = []
    # Load class-index mappings
    with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl', 'rb') as f:
        IDX_TO_CLASS = pickle.load(f)

    index_to_class = {value: key for key, value in IDX_TO_CLASS.items()}

    for face_filename in os.listdir(detected_faces_dir):
        face_path = os.path.join(detected_faces_dir, face_filename)
        face_image = Image.open(face_path)

        face_tensor = transform(face_image).unsqueeze(0).to(device)

        embedding = model(face_tensor).detach().cpu().numpy()

        prediction = clf.predict_proba(embedding)

        max_prob_class = prediction.argmax()

        if prediction[0, max_prob_class] > threshold:
            recognized_name = index_to_class[max_prob_class]
            recognized_faces.append(recognized_name)

    return recognized_faces

def mark_attendance(recognized_faces, attendance_file, subject):
    # today = datetime.date(2023, 5, 28)
    today = datetime.date.today().strftime('%Y-%m-%d')  # Get the current date
    now = datetime.datetime.now().strftime('%H:%M:%S')  # Get the current timestamp

    # Modify the attendance file path based on the subject
    attendance_file = f'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/{subject}_attendance.xlsx'

    if not os.path.exists(attendance_file):
        print(f"Attendance file for subject '{subject}' does not exist. Create a new file for that subject to mark attendance.")
        return
    
    # Read the existing attendance data
    attendance_df = pd.read_excel(attendance_file, index_col=0)

    # Check if attendance for the current day has already been marked
    if today in attendance_df.columns:
        print("Attendance for today has already been marked.")
        return

    # Add the 'Present' column and the timestamp column for the current day
    present_column_name = f'Present_{today}'
    attendance_df[present_column_name] = 0
    attendance_df[f'{today}'] = ''

    # Mark recognized faces as present and update timestamps
    for face in attendance_df.index:
        if face in recognized_faces:
            attendance_df.loc[face, present_column_name] = 1
            attendance_df.loc[face, f'{today}'] = now
        else:
            attendance_df.loc[face, present_column_name] = 0
            attendance_df.loc[face, f'{today}'] = now

    # Calculate the total of the 'Present' column
    total_present = attendance_df[present_column_name].sum()

    # Calculate the total to the 'Present' column and add it to the last cell just after the last entry
    attendance_df.loc['Total', present_column_name] = total_present

    # Calculate the number of present columns
    num_present_columns = attendance_df.filter(like='Present_').shape[1]


    # Calculate the total for each person's row of the 'Present' column
    attendance_df['Total'] = attendance_df.filter(like=f'Present_').sum(axis=1)

    # Calculate attendance percentage which is total of each cell of each row divide by the number of present columns
    attendance_df['Attendance Percentage'] = (attendance_df['Total'] / num_present_columns * 100).map("{:.2f}%".format)
    
    # Save the updated attendance data
    attendance_df.to_excel(attendance_file, index=True)

    print("Marked attendance successfully.")

def process_marking_attendance(subject):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load class-index mappings
    with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/class_to_idx.pkl', 'rb') as f:
        CLASS_TO_IDX = pickle.load(f)
        
    with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl', 'rb') as f:
        IDX_TO_CLASS = pickle.load(f)

    index_to_class = {value: key for key, value in IDX_TO_CLASS.items()}
    names = list(index_to_class.values())

    # Define the paths
    svm_path = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/svm.sav"
    test_data_path = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/testEmbeds.npz"

    # Load the test embeddings and labels
    data = np.load(test_data_path)
    X_test, y_test = data['x'], data['y']

    # Initialize the model structure
    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

    # Load the saved model
    model = model.to(device)

    # Load the saved SVM
    with open(svm_path, 'rb') as file:
        clf = joblib.load(file)

    # Calculate accuracy on test data
    test_acc = accuracy_score(clf.predict(X_test), y_test)
    print(f'Accuracy score on test data: {test_acc:.3f}')

    # Set up face detector
    mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Detect and recognize faces
    detected_faces_dir = 'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces' 
    recognized_faces = recognize_faces(detected_faces_dir, model, clf, transform)
    print(recognized_faces)

    # Example usage
    attendance_file = f'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/{subject}_attendance.xlsx'
    mark_attendance(recognized_faces, attendance_file, subject)

@app.route('/marking-attendance', methods=['POST'])
def marking_attendance():
    subject = request.form.get('subject')
    process_marking_attendance(subject)

    # Return the response as a JSON object
    return jsonify({'message': 'Processed successfully'})

if __name__ == '__main__':
    app.run(host='192.168.137.48', port=5000, debug=True)
