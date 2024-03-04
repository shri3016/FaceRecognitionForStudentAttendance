import datetime
import os
import pickle
import json
import glob
import joblib
import time
import cv2
import shutil
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
import torch
import tqdm
from facenet_pytorch import (MTCNN, InceptionResnetV1,
                             fixed_image_standardization, training)
from PIL import Image
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from mtcnn.mtcnn import MTCNN
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

detected_faces_dir = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces" 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the folder name for storing the detected faces
detected_faces_folder = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces"

# Define your routes here
ABS_PATH="C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/"
DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"

TRAIN_DIR = os.path.join(DATA_PATH, 'train_images')
TEST_DIR = os.path.join(DATA_PATH, 'test_images')

ALIGNED_TRAIN_DIR = os.path.join(DATA_PATH, 'train_images_cropped')
ALIGNED_TEST_DIR = os.path.join(DATA_PATH, 'test_images_cropped')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Folder paths
base_folder = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data"
train_images_folder = os.path.join(base_folder, "train_images")
train_images_cropped_folder = os.path.join(base_folder, "train_images_cropped")
test_images_folder = os.path.join(base_folder, "test_images")
test_images_cropped_folder = os.path.join(base_folder, "test_images_cropped")

# Create folders if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_images_cropped_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)
os.makedirs(test_images_cropped_folder, exist_ok=True)

# Define your functions here
def get_files(path='./', ext=('.png', '.jpeg', '.jpg')):
    """ Get all image files """
    files = []
    for e in ext:
        files.extend(glob.glob(f'{path}/**/*{e}'))
    files.sort(key=lambda p: (os.path.dirname(p), int(os.path.basename(p).split('.')[0])))
    return files

def to_rgb_and_save(path):
    """ Some of the images may have RGBA mode """
    for p in path:
        img = Image.open(p)
        if img.mode != 'RGB':
            img = img.convert('RGB') 
            img.save(p)

def crop_face_and_save(path, new_path=None, model=MTCNN, transformer=None, params=None):
    """
    Detect face on each image, crop them and save to "new_path"
    :param str path: path with images will be passed to  datasets.ImageFolder
    :param str new_path: path to locate new "aligned" images, if new_path is None 
                     then new_path will be path + "_cropped" 
    :param model: model to detect faces, default MTCNN  
    :param transformer: transformer object will be passed to ImageFolder
    :param params: parameters of MTCNN model   
    """
    if not new_path: 
        new_path = path + '_cropped'

    # in case new_path exists MTCNN model will raise error 
    if os.path.exists(new_path):
        shutil.rmtree(new_path)

    # it is default parameters for MTCNN 
    if not params:
        params = {
            'image_size': 160, 'margin': 0, 
            'min_face_size': 10, 'thresholds': [0.6, 0.7, 0.7],
            'factor': 0.709, 'post_process': False, 'device': device
            }
    
    model = model(**params)

    if not transformer:
        transformer = transforms.Lambda(
            lambd=lambda x: x.resize((1280, 1280)) if (np.array(x) > 2000).all() else x
        )
    # for convenience we will use ImageFolder instead of getting Image objects by file paths  
    dataset = datasets.ImageFolder(path, transform=transformer)
    dataset.samples = [(p, p.replace(path, new_path)) for p, _ in dataset.samples]

    # batch size 1 as long as we havent exact image size and MTCNN will raise an error
    loader = DataLoader(dataset, batch_size=1, collate_fn=training.collate_pil)
    for i, (x, y) in enumerate(tqdm.tqdm(loader)): 
        model(x, save_path=y)

    # spare some memory 
    del model, loader, dataset 


def process_train_data(file_path):
    crop_face_and_save(os.path.join(TRAIN_DIR, file_path), os.path.join(ALIGNED_TRAIN_DIR, file_path))

def process_test_data(file_path):
    crop_face_and_save(os.path.join(TEST_DIR, file_path), os.path.join(ALIGNED_TEST_DIR, file_path))

def process_images_in_parallel(src_dir, aligned_dir, process_func):
    file_paths = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    with ThreadPoolExecutor() as executor:
        executor.map(process_func, file_paths)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def fixed_denormalize(image): 
    """ Restandartize images to [0, 255]"""
    return image * 128 + 127.5

def getEmbeds(model, n, loader, imshow=False, n_img=5):
    model.eval()
    # images to display 
    images = []

    embeds, labels = [], []
    for n_img in tqdm.trange(n): 
        for i, (x, y) in enumerate(loader, 1): 

            # on each first batch get 'n_img' images  
            if imshow and i == 1: 
                inds = np.random.choice(x.size(0), min(x.size(0), n_img))
                images.append(fixed_denormalize(x[inds].data.cpu()).permute((0, 2, 3, 1)).numpy())

            embed = model(x.to(device))
            embed = embed.data.cpu().numpy()
            embeds.append(embed), labels.extend(y.data.cpu().numpy())

    return np.concatenate(embeds), np.array(labels)

def process_training_script():

    # 1. Get path for TRAIN_DIR/TEST_DIR
    trainF, testF = get_files(TRAIN_DIR), get_files(TEST_DIR)

    # prepare info for printing
    trainC, testC = Counter(map(os.path.dirname, trainF)), Counter(map(os.path.dirname, testF))
    train_total, train_text  = sum(trainC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in trainC.items()])
    test_total, test_text  = sum(testC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in testC.items()])

    print(f'Train files\n\tpath: {TRAIN_DIR}\n\ttotal number: {train_total}\n{train_text}')
    print(f'Test files\n\tpath: {TEST_DIR}\n\ttotal number: {test_total}\n{test_text}')

    to_rgb_and_save(trainF), to_rgb_and_save(testF)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    print("Whether CUDA is supported by our system:", torch.cuda.is_available())

    if not os.path.exists(ALIGNED_TRAIN_DIR):
        os.makedirs(ALIGNED_TRAIN_DIR)

    if not os.path.exists(ALIGNED_TEST_DIR):
        os.makedirs(ALIGNED_TEST_DIR)

    # 3. Crop train dataset faces and save aligned images 
    # print('\t- Train data')
    process_images_in_parallel(TRAIN_DIR, ALIGNED_TRAIN_DIR, process_train_data)

    # Crop test dataset faces and save aligned images 
    # print('\t- Test data')
    process_images_in_parallel(TEST_DIR, ALIGNED_TEST_DIR, process_test_data)

    # Check if some imgs were missed by detector and failed to save 
    train_files, train_aligned_files = get_files(TRAIN_DIR), get_files(ALIGNED_TRAIN_DIR)
    test_files, test_aligned_files = get_files(TEST_DIR), get_files(ALIGNED_TEST_DIR)

    for dataset_type, src_files, aligned_files, src_dir, aligned_dir in [("Train", train_files, train_aligned_files, TRAIN_DIR, ALIGNED_TRAIN_DIR),
                                                                        ("Test", test_files, test_aligned_files, TEST_DIR, ALIGNED_TEST_DIR)]:
        if len(src_files) != len(aligned_files): 
            files = set(map(lambda fp: os.path.relpath(fp, start=src_dir), src_files))
            aligned_files = set(map(lambda fp: os.path.relpath(fp, start=aligned_dir), aligned_files))
            detect_failed_files = list(files - aligned_files)
            print(f"\n{dataset_type} dataset: {len(aligned_files)}/{len(files)} files were not saved: {', '.join(detect_failed_files)}")
            
            if dataset_type == "Train":
                trainFailF = list(map(lambda fp: os.path.join(TRAIN_DIR, fp), detect_failed_files))

    trainF = get_files(ALIGNED_TRAIN_DIR)
    testF = get_files(ALIGNED_TEST_DIR)

    standard_transform = transforms.Compose([
                                    np.float32, 
                                    transforms.ToTensor(),
                                    fixed_image_standardization
    ])

    aug_mask = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.15),
                    A.RandomBrightnessContrast(contrast_limit=0.5, p=0.4),
                    A.Rotate(30, p=0.2),
                    A.RandomSizedCrop((120, 120), 160, 160, p=0.4),
                    A.OneOrOther(A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2), A.Blur(p=0.2), p=0.66),
                    A.OneOf([
                                A.Rotate(45, p=0.3),
                                A.ElasticTransform(sigma=20, alpha_affine=20, border_mode=0, p=0.2)
                                ], p=0.5),
                    A.HueSaturationValue(val_shift_limit=10, p=0.3)
    ], p=1)

    transform = {
        'train': transforms.Compose([
                                    transforms.Lambda(lambd=lambda x: aug_mask(image=np.array(x))['image']),
                                    standard_transform
        ]),
        'test': standard_transform
    }


    b = 32

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Original train images
    trainD = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=test_transform)
    # Augmented train images
    trainD_aug = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=train_transform)
    # Train Loader
    trainL = DataLoader(trainD, batch_size=b, num_workers=0)
    trainL_aug = DataLoader(trainD_aug, batch_size=b, num_workers=0)

    # Original test images
    testD = datasets.ImageFolder(ALIGNED_TEST_DIR, transform=test_transform)
    # Test Loader
    testL = DataLoader(testD, batch_size=b, num_workers=0)

    # Convert encoded labels to named classes
    IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
    CLASS_TO_IDX = dict(trainD.class_to_idx.items())

    # Save IDX_TO_CLASS
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/class_to_idx.pkl", 'wb') as f:
        pickle.dump(IDX_TO_CLASS, f)

    # Save CLASS_TO_IDX
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl", 'wb') as f:
        pickle.dump(CLASS_TO_IDX, f)

    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, device=device).eval()

    # Get embeddings
    trainEmbeds, trainLabels = getEmbeds(model, 1, trainL, False)
    trainEmbeds_aug, trainLabels_aug = getEmbeds(model, 50, trainL_aug, imshow=True, n_img=3)

    trainEmbeds = np.concatenate([trainEmbeds, trainEmbeds_aug])
    trainLabels = np.concatenate([trainLabels, trainLabels_aug])

    # Test embeddings
    testEmbeds, testLabels = getEmbeds(model, 1, testL, False)

    DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data"

    # 4. Save embeddings 
    TRAIN_EMBEDS = os.path.join(DATA_PATH, 'trainEmbeds.npz')
    TEST_EMBEDS = os.path.join(DATA_PATH, 'testEmbeds.npz')

    np.savez(TRAIN_EMBEDS, x=trainEmbeds, y=trainLabels)
    np.savez(TEST_EMBEDS, x=testEmbeds, y=testLabels)

    # Load the saved embeddings to use them futher 
    trainEmbeds, trainLabels = np.load(TRAIN_EMBEDS, allow_pickle=True).values()
    testEmbeds, testLabels = np.load(TEST_EMBEDS, allow_pickle=True).values()

    # Get named labels
    trainLabels, testLabels = IDX_TO_CLASS[trainLabels], IDX_TO_CLASS[testLabels]

    # data preparation 
    X = np.copy(trainEmbeds)
    y = np.array([CLASS_TO_IDX[label] for label in trainLabels])

    print(f'X train embeds size: {X.shape}')
    print(f'Target train size: {y.shape}')
    print("Started Time part now...")
    warnings.filterwarnings('ignore', 'Solver terminated early.*')

    param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'auto'],
                  'kernel': ['rbf', 'sigmoid', 'poly']}
    model_params = {'class_weight': 'balanced', 'max_iter': 10, 'probability': True, 'random_state': 3}
    model = SVC(**model_params)
    clf = GridSearchCV(model, param_grid)
    # clf.fit(X, y)

    start_time = time.time()
    print("Started fitting now...")
    clf.fit(X, y)
    print("Finished fitting now...")
    end_time = time.time()
    execution_time = end_time - start_time

    print('Execution time:', execution_time, 'seconds')
    print("Finished Time part now...")

    DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data"
    SVM_PATH = os.path.join(DATA_PATH, 'svm.sav')

    # Save SVC model
    joblib.dump(clf, SVM_PATH)

    return "Training Script executed successfully"

# Recognize faces
def recognize_faces(detected_faces_dir, model, clf, transform, threshold=0.5):
    recognized_faces = []
    # Load class-index mappings
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl", 'rb') as f:
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

def create_attendance_file(attendance_file, subject):
    # Load class-index mappings
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl", 'rb') as f:
        IDX_TO_CLASS = pickle.load(f)

    index_to_class = {value: key for key, value in IDX_TO_CLASS.items()}
    names = list(index_to_class.values())

    # Check if the attendance file already exists
    if os.path.exists(attendance_file):
        # Delete the existing attendance file
        os.remove(attendance_file)
        print("Existing attendance file deleted.")

    # Get the student names from the names variable
    student_list = names

    # Create a DataFrame with student names under the 'Name' column
    attendance_df = pd.DataFrame(student_list, columns=['Name'])

    # Modify the attendance file path based on the subject
    attendance_file = f'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/{subject}_attendance.xlsx'

    # Save the DataFrame to an Excel file
    attendance_df.to_excel(attendance_file, index=False)

    print("Attendance file created successfully.")

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

def process_creating_attendance_file(subject):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load class-index mappings
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/class_to_idx.pkl", 'rb') as f:
        CLASS_TO_IDX = pickle.load(f)
        
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl", 'rb') as f:
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

    # # Set up face detector
    # mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Detect and recognize faces
    detected_faces_dir = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces" 
    recognized_faces = recognize_faces(detected_faces_dir, model, clf, transform)
    print(recognized_faces)

    # Example usage
    attendance_file = f'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/{subject}_attendance.xlsx'
    create_attendance_file(attendance_file, subject)

    attendance_file = f'C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/{subject}_attendance.xlsx'
    mark_attendance(recognized_faces, attendance_file, subject)

def process_marking_attendance(subject):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load class-index mappings
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/class_to_idx.pkl", 'rb') as f:
        CLASS_TO_IDX = pickle.load(f)
        
    with open("C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl", 'rb') as f:
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

    # # Set up face detector
    # mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Detect and recognize faces
    detected_faces_dir = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/detected_faces" 
    recognized_faces = recognize_faces(detected_faces_dir, model, clf, transform)
    print(recognized_faces)

    # Example usage
    attendance_file = f"C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/{subject}_attendance.xlsx"
    mark_attendance(recognized_faces, attendance_file, subject)

def process_teachers_detecting_faces():
    # Check if the detected_faces folder already exists
    if os.path.exists(detected_faces_folder):
        # If the folder exists, delete it and its contents
        shutil.rmtree(detected_faces_folder)

    # Create a new folder to store the detected faces
    os.makedirs(detected_faces_folder)

    # Create the detector, using default weights
    detector = MTCNN()

    # Open the default camera (ID=0) or RTSP link (for cctv camera)
    # cap = cv2.VideoCapture("rtsp://192.168.72.211:554/user=admin_password=IcRKgw5I_channel=0_stream=0.sdp?real_stream")
    cap = cv2.VideoCapture(0)

    # Counter to track the number of saved faces
    face_counter = 0

    while True:
        # Read a frame from the camera stream
        ret, img = cap.read()

        # Resize the frame to a smaller size
        img = cv2.resize(img, (1535,790))

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

    return "Detected faces Script executed successfully"

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
    # cap = cv2.VideoCapture("rtsp://192.168.21.211:554/user=admin_password=8M0sJUeP_channel=0_stream=0.sdp?real_stream")
    cap = cv2.VideoCapture(0)

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
        "message": f"Train folders for student: {name} was made successfully"
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
    # cap = cv2.VideoCapture("rtsp://192.168.21.211:554/user=admin_password=8M0sJUeP_channel=0_stream=0.sdp?real_stream")
    cap = cv2.VideoCapture(0)

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
        "message": f"Test folders for student: {name} was made successfully"
    }
    return json.dumps(result2)

@app.route('/admin-training', methods=['POST'])
def admin_training():
    process_training_script()

    # Return the response as a JSON object
    return jsonify({'message': 'Training Complete'})

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

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/teachers-detecting-faces', methods=['POST'])
def teachers_detecting_faces():
    process_teachers_detecting_faces()

    # Return the response as a JSON object
    return jsonify({'message': 'Detected faces successfully'})

@app.route('/creating-attendance-file', methods=['POST'])
def creating_attendance_file():
    subject = request.form.get('subject')
    process_creating_attendance_file(subject)

    # Return the response as a JSON object
    return jsonify({'message': f'Attendance file for {subject} was created successfully. Marked attendance as well was successful'})

@app.route('/marking-attendance', methods=['POST'])
def marking_attendance():
    subject = request.form.get('subject')
    process_marking_attendance(subject)

    # Return the response as a JSON object
    return jsonify({'message': 'Marked attendance successfully'})

if __name__ == '__main__':
    app.run(host='192.168.72.6', port=5000, debug=True)