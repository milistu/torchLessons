import os
import random
import urllib.request
import tarfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import imutils
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mping # TODO: Use only one lib for image viz, TURBO JPEG

import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

def getData():
    print(f"[INFO] Current Working Directory: {os.getcwd()}")

    if not os.path.exists('data/ibug_300W_large_face_landmark_dataset'):
        url = 'http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz'
        file_name = 'data/ibug_300W_large_face_landmark_dataset.tar.gz'

        print("[INFO] Downloading the dataset")
        urllib.request.urlretrieve(url, file_name)

        with tarfile.open(file_name, 'r:gz') as tar:
            print("[INFO] Extracting the dataset")
            tar.extractall('data/')

        print("[INFO] Removing gzip")
        os.remove(file_name)
    else:
        print(f"[INFO] Dataset already present")

def visualizeData():
    file = open("data/ibug_300W_large_face_landmark_dataset/helen/trainset/100032540_1.pts")
    points = file.readlines()[3:-1]

    landmarks = []

    for point in points:
        x,y = point.split(' ')
        landmarks.append([np.floor(float(x)), np.floor(float(y[:-1]))])
    
    landmarks = np.array(landmarks)

    plt.figure(figsize=(10, 10))
    plt.imshow(mping.imread('data/ibug_300W_large_face_landmark_dataset/helen/trainset/100032540_1.jpg'))
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 5, c = 'g')
    plt.show()

def visualizeTransforms():
    dataset = FaceLandmarksDataset(Transforms())

    image, landmarks = dataset[0]
    landmarks = (landmarks + 0.5) * 224
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 8)
    plt.show()
    
class Transforms():
    def __init__(self): pass

    def rotate(self, image, landmarks, angle):
        '''Randomly rotate the face.'''
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5

        return Image.fromarray(image), new_landmarks
    
    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks
    
    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks
    
    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]]) 

        return image, landmarks
    
    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])

        return image, landmarks
    
class FaceLandmarksDataset(Dataset):
    
    def __init__(self, transform=None):

        tree = ET.parse('data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'data/ibug_300W_large_face_landmark_dataset'

        # Acces data info 
        for filename in root[2]:
            # Get image paths
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
            # Get crop data
            self.crops.append(filename[0].attrib)
            # Get landmark data
            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks, dtype=np.float32)

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])
        
        # Zero-centre the landmarks - easier for NN to learn
        landmarks = landmarks - 0.5

        return image, landmarks


if __name__ == '__main__':
    getData()
    visualizeData()
    visualizeTransforms()
    # Add data example

