import torch
from torch.utils.data import DataLoader, Dataset, random_split

import data
import sys

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write(f"Train Steps: {step/total_step}  Loss: {loss:.4f} ")
    else:
        sys.stdout.write(f"Valid Steps: {step/total_step}  Loss: {loss:.4f} ")
    
    sys.stdout.flush()

# Create the Dataset
dataset = data.FaceLandmarksDataset(data.Transforms())

# Split the dataset into validation and test sets
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set

print(f"[INFO] The length of Train set is: {len_train_set}")
print(f"[INFO] The length of Valid set is: {len_valid_set}")

train_dataset, valid_dataset = random_split(dataset, [len_train_set, len_valid_set])

# Shuffle and batch the datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

# Testing the shape of input data
images, landmarks = next(iter(train_loader))

print(f"Test images size: {images.shape}")
print(f"Test landmarks size: {landmarks.shape}")



