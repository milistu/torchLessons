import os
import random
import urllib.request
import tarfile

import numpy as np
import imutils
from PIL import Image
import torch

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



if __name__ == '__main__':
    getData()


