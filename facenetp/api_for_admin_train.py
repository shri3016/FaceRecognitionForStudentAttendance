import glob
import joblib
import os
import pickle
import shutil
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
import numpy as np
import torch
import tqdm
from facenet_pytorch import (MTCNN, InceptionResnetV1,
                             fixed_image_standardization, training)
from PIL import Image
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from torchvision import datasets, transforms
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define your routes here
ABS_PATH="C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/"
DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"

TRAIN_DIR = os.path.join(DATA_PATH, 'train_images')
TEST_DIR = os.path.join(DATA_PATH, 'test_images')

ALIGNED_TRAIN_DIR = os.path.join(DATA_PATH, 'train_images_cropped')
ALIGNED_TEST_DIR = os.path.join(DATA_PATH, 'test_images_cropped')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# # 3. Crop train dataset faces and save aligned images 
# print('\t- Train data')
# process_images_in_parallel(TRAIN_DIR, ALIGNED_TRAIN_DIR, process_train_data)

# # Crop test dataset faces and save aligned images 
# print('\t- Test data')
# process_images_in_parallel(TEST_DIR, ALIGNED_TEST_DIR, process_test_data)

# # Check if some imgs were missed by detector and failed to save 
# train_files, train_aligned_files = get_files(TRAIN_DIR), get_files(ALIGNED_TRAIN_DIR)
# test_files, test_aligned_files = get_files(TEST_DIR), get_files(ALIGNED_TEST_DIR)

# for dataset_type, src_files, aligned_files, src_dir, aligned_dir in [("Train", train_files, train_aligned_files, TRAIN_DIR, ALIGNED_TRAIN_DIR),
#                                                                       ("Test", test_files, test_aligned_files, TEST_DIR, ALIGNED_TEST_DIR)]:
#     if len(src_files) != len(aligned_files): 
#         files = set(map(lambda fp: os.path.relpath(fp, start=src_dir), src_files))
#         aligned_files = set(map(lambda fp: os.path.relpath(fp, start=aligned_dir), aligned_files))
#         detect_failed_files = list(files - aligned_files)
#         print(f"\n{dataset_type} dataset: {len(aligned_files)}/{len(files)} files were not saved: {', '.join(detect_failed_files)}")
        
#         if dataset_type == "Train":
#             trainFailF = list(map(lambda fp: os.path.join(TRAIN_DIR, fp), detect_failed_files))
#             # plot(paths=trainFailF)

# trainF = get_files(ALIGNED_TRAIN_DIR)
# testF = get_files(ALIGNED_TEST_DIR)

# standard_transform = transforms.Compose([
#                                 np.float32, 
#                                 transforms.ToTensor(),
#                                 fixed_image_standardization
# ])

# aug_mask = A.Compose([
#                    A.HorizontalFlip(p=0.5),
#                    A.VerticalFlip(p=0.15),
#                    A.RandomBrightnessContrast(contrast_limit=0.5, p=0.4),
#                    A.Rotate(30, p=0.2),
#                    A.RandomSizedCrop((120, 120), 160, 160, p=0.4),
#                    A.OneOrOther(A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2), A.Blur(p=0.2), p=0.66),
#                    A.OneOf([
#                             A.Rotate(45, p=0.3),
#                             A.ElasticTransform(sigma=20, alpha_affine=20, border_mode=0, p=0.2)
#                             ], p=0.5),
#                   A.HueSaturationValue(val_shift_limit=10, p=0.3)
# ], p=1)

# transform = {
#     'train': transforms.Compose([
#                                  transforms.Lambda(lambd=lambda x: aug_mask(image=np.array(x))['image']),
#                                  standard_transform
#     ]),
#     'test': standard_transform
# }


# b = 32

# train_transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# # Original train images
# trainD = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=test_transform)
# # Augmented train images
# trainD_aug = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=train_transform)
# # Train Loader
# trainL = DataLoader(trainD, batch_size=b, num_workers=0)
# trainL_aug = DataLoader(trainD_aug, batch_size=b, num_workers=0)

# # Original test images
# testD = datasets.ImageFolder(ALIGNED_TEST_DIR, transform=test_transform)
# # Test Loader
# testL = DataLoader(testD, batch_size=b, num_workers=0)

# # Convert encoded labels to named classes
# IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
# CLASS_TO_IDX = dict(trainD.class_to_idx.items())

# # Save IDX_TO_CLASS
# with open('C:/Users/natha/Desktop/image_preprocessing/facenetp/data/class_to_idx.pkl', 'wb') as f:
#     pickle.dump(IDX_TO_CLASS, f)

# # Save CLASS_TO_IDX
# with open('C:/Users/natha/Desktop/image_preprocessing/facenetp/data/idx_to_class.pkl', 'wb') as f:
#     pickle.dump(CLASS_TO_IDX, f)

# model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, device=device).eval()

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

# # Get embeddings
# trainEmbeds, trainLabels = getEmbeds(model, 1, trainL, False)
# trainEmbeds_aug, trainLabels_aug = getEmbeds(model, 50, trainL_aug, imshow=True, n_img=3)

# trainEmbeds = np.concatenate([trainEmbeds, trainEmbeds_aug])
# trainLabels = np.concatenate([trainLabels, trainLabels_aug])

# # Test embeddings
# testEmbeds, testLabels = getEmbeds(model, 1, testL, False)

# # 4. Save embeddings 
# TRAIN_EMBEDS = os.path.join(DATA_PATH, 'trainEmbeds.npz')
# TEST_EMBEDS = os.path.join(DATA_PATH, 'testEmbeds.npz')

# np.savez(TRAIN_EMBEDS, x=trainEmbeds, y=trainLabels)
# np.savez(TEST_EMBEDS, x=testEmbeds, y=testLabels)

# # Load the saved embeddings to use them futher 
# trainEmbeds, trainLabels = np.load(TRAIN_EMBEDS, allow_pickle=True).values()
# testEmbeds, testLabels = np.load(TEST_EMBEDS, allow_pickle=True).values()

# # Get named labels
# trainLabels, testLabels = IDX_TO_CLASS[trainLabels], IDX_TO_CLASS[testLabels]

# # data preparation 
# X = np.copy(trainEmbeds)
# y = np.array([CLASS_TO_IDX[label] for label in trainLabels])

# print(f'X train embeds size: {X.shape}')
# print(f'Tagret train size: {y.shape}')


# warnings.filterwarnings('ignore', 'Solver terminated early.*')

# param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'auto'],
#               'kernel': ['rbf', 'sigmoid', 'poly']}

# model_params = {'class_weight': 'balanced', 'max_iter': 10, 'probability': True, 'random_state': 3}

# model = SVC(**model_params)

# clf = GridSearchCV(model, param_grid)

# start_time = time.time()
# print("Started fitting now.")
# clf.fit(X, y)
# print("Finished fitting now.")
# end_time = time.time()

# execution_time = end_time - start_time

# print('Best estimator:', clf.best_estimator_)
# print('Best params:', clf.best_params_)
# print('Execution time:', execution_time, 'seconds')
# print("Finished Time part now.")

# DATA_PATH = "C:/Users/natha/Desktop/image_preprocessing/facenetp/data"
# SVM_PATH = os.path.join(DATA_PATH, 'svm.sav')
# MODEL_PATH = os.path.join(DATA_PATH, 'facenet_pytorch_model.pth')

# # Save SVC model
# joblib.dump(clf, SVM_PATH)

# # Save PyTorch model
# torch.save(model.state_dict(), MODEL_PATH)

def process_training_script():

    # 1. Get path for TRAIN_DIR/TEST_DIR
    trainF, testF = get_files(TRAIN_DIR), get_files(TEST_DIR)

    # prepare info for printing
    trainC, testC = Counter(map(os.path.dirname, trainF)), Counter(map(os.path.dirname, testF))
    train_total, train_text  = sum(trainC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in trainC.items()])
    test_total, test_text  = sum(testC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in testC.items()])

    # print(f'Train files\n\tpath: {TRAIN_DIR}\n\ttotal number: {train_total}\n{train_text}')
    # print(f'Test files\n\tpath: {TEST_DIR}\n\ttotal number: {test_total}\n{test_text}')

    to_rgb_and_save(trainF), to_rgb_and_save(testF)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f'Running on device: {device}')
    # print("Whether CUDA is supported by our system:", torch.cuda.is_available())

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
    with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/class_to_idx.pkl', 'wb') as f:
        pickle.dump(IDX_TO_CLASS, f)

    # Save CLASS_TO_IDX
    with open('C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/idx_to_class.pkl', 'wb') as f:
        pickle.dump(CLASS_TO_IDX, f)

    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, device=device).eval()

    # Get embeddings
    trainEmbeds, trainLabels = getEmbeds(model, 1, trainL, False)
    trainEmbeds_aug, trainLabels_aug = getEmbeds(model, 50, trainL_aug, imshow=True, n_img=3)

    trainEmbeds = np.concatenate([trainEmbeds, trainEmbeds_aug])
    trainLabels = np.concatenate([trainLabels, trainLabels_aug])

    # Test embeddings
    testEmbeds, testLabels = getEmbeds(model, 1, testL, False)

    DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"

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

    warnings.filterwarnings('ignore', 'Solver terminated early.*')

    param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'auto'],
                  'kernel': ['rbf', 'sigmoid', 'poly']}
    model_params = {'class_weight': 'balanced', 'max_iter': 10, 'probability': True, 'random_state': 3}
    model = SVC(**model_params)
    clf = GridSearchCV(model, param_grid)
    # clf.fit(X, y)

    start_time = time.time()
    print("Started fitting now.")
    clf.fit(X, y)
    print("Finished fitting now.")
    end_time = time.time()
    execution_time = end_time - start_time

    print('Execution time:', execution_time, 'seconds')
    print("Finished Time part now.")

    DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"
    SVM_PATH = os.path.join(DATA_PATH, 'svm.sav')

    # Save SVC model
    joblib.dump(clf, SVM_PATH)

    return "Training Script executed successfully"

@app.route('/admin-training', methods=['POST'])
def admin_training():
    process_training_script()

    # Return the response as a JSON object
    return jsonify({'message': 'Processed successfully'})

if __name__ == '__main__':
    app.run(debug=True)