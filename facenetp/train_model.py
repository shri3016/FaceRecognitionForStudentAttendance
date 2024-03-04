import glob
import joblib
import os
import pickle
import shutil
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import ceil
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from facenet_pytorch import (MTCNN, InceptionResnetV1,
                             fixed_image_standardization, training)
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image, ImageDraw
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from flask import Flask, jsonify

app = Flask(__name__)

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

@ app.route('/train_model', methods=['GET'])
def train_model():
    ABS_PATH="C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/"
    DATA_PATH = "C:/Users/91820/OneDrive/Desktop/FYProject/facenetp/data/"

    TRAIN_DIR = os.path.join(DATA_PATH, 'train_images')
    TEST_DIR = os.path.join(DATA_PATH, 'test_images')

    ALIGNED_TRAIN_DIR = os.path.join(DATA_PATH, 'train_images_cropped')
    ALIGNED_TEST_DIR = os.path.join(DATA_PATH, 'test_images_cropped')

    # Get path for TRAIN_DIR/TEST_DIR
    trainF, testF = get_files(TRAIN_DIR), get_files(TEST_DIR)

    # prepare info for printing
    trainC, testC = Counter(map(os.path.dirname, trainF)), Counter(map(os.path.dirname, testF))
    train_total, train_text  = sum(trainC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in trainC.items()])
    test_total, test_text  = sum(testC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in testC.items()])

    print(f'Train files\n\tpath: {TRAIN_DIR}\n\ttotal number: {train_total}\n{train_text}')
    print(f'Test files\n\tpath: {TEST_DIR}\n\ttotal number: {test_total}\n{test_text}')

    to_rgb_and_save(trainF), to_rgb_and_save(testF)

    # def imshow(img, ax, title):  
    #     ax.imshow(img)
    #     if title:
    #         el = Ellipse((2, -1), 0.5, 0.5)
    #         ax.annotate(title, xy=(1, 0), xycoords='axes fraction', ha='right', va='bottom',
    #                     bbox=dict(boxstyle="round", fc="0.8"), 
    #                     arrowprops=dict(arrowstyle="->",
    #                                     connectionstyle="angle,angleA=0,angleB=90,rad=10"),
    #                     )
    #     ax.axis('off')

    # def show_pictures(fns, titles=None, per_row=5, axsize=(2.5, 2.5)):
    #     fns = fns[:per_row**2] # showing 5x5 images
    #     titles = titles[:per_row**2] if titles else [None]*len(fns)
    #     rows = len(fns) // per_row + (1 if len(fns) % per_row else 0)
    #     fig = plt.figure(figsize=(per_row * axsize[0], rows * axsize[1])) # (width, height)
    #     grid = ImageGrid(fig, 111, nrows_ncols=(rows, per_row), axes_pad=0.1)
    #     for ax, fp, title in zip(grid, fns, titles):
    #         img = Image.open(fp)
    #         imshow(img, ax, title)
    #     plt.show()

    # show_pictures(trainF, per_row=10, axsize=(2,2))

    # Initialize MTCNN and InceptionResnetV1 models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def collate_fn(x):
        return x[0]

    def get_embedding(img_path):
        img = Image.open(img_path)
        img_cropped = mtcnn(img)
        img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
        return img_embedding.detach().cpu()

    def get_aligned_faces(src_dir, dest_dir):
        """Detect and align faces in images using MTCNN"""
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for img_path in tqdm.tqdm(get_files(src_dir)):
            dest_path = os.path.join(dest_dir, os.path.relpath(img_path, src_dir))
            dest_subdir = os.path.dirname(dest_path)
            if not os.path.exists(dest_subdir):
                os.makedirs(dest_subdir)
            try:
                img = Image.open(img_path)
                mtcnn(img, save_path=dest_path)
            except Exception as e:
                print(f'Error processing {img_path}: {e}')
                continue

    # Align faces in the training and test sets
    get_aligned_faces(TRAIN_DIR, ALIGNED_TRAIN_DIR)
    get_aligned_faces(TEST_DIR, ALIGNED_TEST_DIR)

    # Create the training dataset
    train_dataset = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=transforms.Resize((256, 256)))
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32, num_workers=4)

    # Calculate image embeddings for the training set
    train_embeddings = []
    for img, _ in tqdm.tqdm(train_loader):
        img_embedding = resnet(img.to(device)).detach().cpu()
        train_embeddings.extend(img_embedding)

    # Convert embeddings to NumPy array
    train_embeddings = torch.stack(train_embeddings).numpy()

    # Create the test dataset
    test_dataset = datasets.ImageFolder(ALIGNED_TEST_DIR, transform=transforms.Resize((256, 256)))
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=32, num_workers=4)

    # Calculate image embeddings for the test set
    test_embeddings = []
    for img, _ in tqdm.tqdm(test_loader):
        img_embedding = resnet(img.to(device)).detach().cpu()
        test_embeddings.extend(img_embedding)

    # Convert embeddings to NumPy array
    test_embeddings = torch.stack(test_embeddings).numpy()

    # Train SVM classifier
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'kernel': ['rbf']}
    svm = SVC(probability=True)
    svm_grid = RandomizedSearchCV(svm, param_grid, n_jobs=-1, verbose=1, cv=5)
    svm_grid.fit(train_embeddings, train_dataset.targets)

    # Save the trained SVM model
    model_path = os.path.join(ABS_PATH, 'svm_model.pkl')
    joblib.dump(svm_grid.best_estimator_, model_path)

    # Calculate accuracy on the test set
    accuracy = svm_grid.score(test_embeddings, test_dataset.targets)

    response = {
        'message': 'Model training completed!',
        'accuracy': accuracy,
        'model_path': model_path
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
