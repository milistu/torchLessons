import os
import random
import urllib.request
import tarfile

import numpy as np
import imutils
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms

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
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]]) # ???

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

if __name__ == '__main__':
    getData()


