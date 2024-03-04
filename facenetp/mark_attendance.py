import datetime
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics import accuracy_score
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load class-index mappings
with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/class_to_idx.pkl', 'rb') as f:
    CLASS_TO_IDX = pickle.load(f)
    
with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl', 'rb') as f:
    IDX_TO_CLASS = pickle.load(f)

index_to_class = {value: key for key, value in IDX_TO_CLASS.items()}
names = list(index_to_class.values())

# Define the paths
# model_path = "path_to_your_model/model.pth"
svm_path = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/svm.sav"
test_data_path = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/testEmbeds.npz"

# Load the test embeddings and labels
data = np.load(test_data_path)
X_test, y_test = data['x'], data['y']

# Initialize the model structure
model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

# Load the saved model
# model.load_state_dict(torch.load(model_path))
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

# Recognize faces
def recognize_faces(detected_faces_dir, model, clf, transform, threshold=0.5):
    recognized_faces = []

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

# Detect and recognize faces
detected_faces_dir = 'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces' 
recognized_faces = recognize_faces(detected_faces_dir, model, clf, transform)
print(recognized_faces)

def mark_attendance(recognized_faces, attendance_file):
    today = datetime.date.today().strftime('%Y-%m-%d')
    now = datetime.datetime.now().strftime('%H:%M:%S')  # Get the current timestamp

    if not os.path.exists(attendance_file):
        columns = ['Name']
        data = []  # Initialize an empty list

        # Get the student names from the names variable
        student_list = names

        for name in student_list:
            name = name.strip()  # Remove any leading/trailing whitespace
            if name != "":
                data.append([name])  # Append the name as a list to the data list

        attendance_df = pd.DataFrame(data, columns=columns)
        attendance_df.to_excel(attendance_file, index=False)

        print("Attendance file created successfully.")

    # Read the existing attendance data
    attendance_df = pd.read_excel(attendance_file, index_col=0)

    # Add the date column if it doesn't exist
    if today not in attendance_df.columns:
        attendance_df[f'Present'] = 0
        attendance_df[f'{today}'] = ''

    # Mark recognized faces as present and update timestamps
    for face in attendance_df.index:
        # print(face)
        if face in recognized_faces:
            attendance_df.loc[face, 'Present'] = 1
            attendance_df.loc[face, f'{today}'] = now
        else:
            attendance_df.loc[face, 'Present'] = 0
            attendance_df.loc[face, f'{today}'] = now

    # Calculate the total of the 'Present' column
    total_present = attendance_df[f'Present'].sum()

    # Calculate the total to the 'Present' column and add it to the last cell just after the last entry
    attendance_df.loc['Total', f'Present'] = total_present

    # Calculate the total for each person's row of the 'Present' column
    attendance_df['Total'] = attendance_df.filter(like='Present').sum(axis=1)

    # Save the updated attendance data
    attendance_df.to_excel(attendance_file, index=True)
    
    print("Marked attendance successfully.")

# Example usage
attendance_file = 'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/attendance.xlsx'
mark_attendance(recognized_faces, attendance_file)


# def mark_attendance(recognized_faces, attendance_file, attendance_date):
#     attendance_date = attendance_date.strftime('%Y-%m-%d')
#     now = datetime.datetime.now().strftime('%H:%M:%S')  # Get the current timestamp

#     if not os.path.exists(attendance_file):
#         # Create a new attendance file with 'Name' column
#         columns = ['Name']
#         data = []

#         student_list = names
#         for name in student_list:
#             name = name.strip()
#             if name != "":
#                 data.append([name])

#         attendance_df = pd.DataFrame(data, columns=columns)
#         attendance_df.to_excel(attendance_file, index=False)

#         print("Attendance file created successfully.")

#     # Read the existing attendance data
#     attendance_df = pd.read_excel(attendance_file, index_col=0)

#     # Check if the date already exists
#     if attendance_date in attendance_df.columns:
#         print(f"The date {attendance_date} already exists in the attendance file. Please input another date.")
#         return

#     # Add the date column
#     attendance_df.insert(len(attendance_df.columns), attendance_date, '')
#     attendance_df.insert(len(attendance_df.columns), attendance_date + "_Present", 0)

#     # Mark recognized faces as present and update timestamps
#     for face in attendance_df.index:
#         if face in recognized_faces:
#             attendance_df.loc[face, attendance_date + "_Present"] = 1
#         attendance_df.loc[face, attendance_date] = now

#     # Check if 'Total' column exists
#     if 'Total' not in attendance_df.columns:
#         # Create 'Total' column with initial values set to 0
#         attendance_df['Total'] = 0

#     # Reorder the columns
#     columns = ['Name'] + [col for col in attendance_df.columns if col != 'Name' and col != 'Total'] + ['Total']
#     attendance_df = attendance_df[columns]

#     # Calculate the total of the 'Present' column for each row
#     attendance_df['Total'] = attendance_df.filter(like='_Present').sum(axis=1)

#     # Save the updated attendance data
#     attendance_df.to_excel(attendance_file, index=True)

#     print("Marked attendance successfully.")

# Example usage
# attendance_file = 'C:/Users/natha/Desktop/image_preprocessing/facenetp/attendance.xlsx'
# attendance_date = datetime.date(2023, 5, 28)  # Example date
# mark_attendance(recognized_faces, attendance_file, attendance_date)
# attendance_file = 'C:/Users/natha/Desktop/image_preprocessing/facenetp/attendance.xlsx'
# mark_attendance(recognized_faces, attendance_file)