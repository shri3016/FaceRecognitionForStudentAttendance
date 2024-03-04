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

def create_attendance_file(attendance_file):
    # Check if the attendance file already exists
    if os.path.exists(attendance_file):
        # Delete the existing attendance file
        os.remove(attendance_file)
        print("Existing attendance file deleted.")

    # Get the student names from the names variable
    student_list = names

    # Create a DataFrame with student names under the 'Name' column
    attendance_df = pd.DataFrame(student_list, columns=['Names'])

    # Save the DataFrame to an Excel file
    attendance_df.to_excel(attendance_file, index=False)

    print("Attendance file created successfully.")

# Call the function to create the attendance file
attendance_file = 'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/attendance.xlsx' # Replace with the path to your attendance Excel file
create_attendance_file(attendance_file)