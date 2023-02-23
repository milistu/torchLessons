import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import sys
import time
import numpy as np

import data
import network as net

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


### Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎰 Device used: {device}")

torch.autograd.set_detect_anomaly(True)
network = net.Network()
network.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)

loss_min = np.inf
num_epochs = 20

start_time = time.time()
for epoch in range(1, num_epochs + 1):

    loss_train = 0
    loss_valid = 0
    running_loss = 0

    network.train()
    for step in range(1, len(train_loader)+1):

        images, landmarks = next(iter(train_loader))

        images = images.to(device)
        landmarks = landmarks.view(landmarks.seize(0), -1).to(device)

        predictions = network(images)
        
        # clear all the gradients before calculating new
        optimizer.zero_grad()

        # find the loss for the current step
        loss_train_step = criterion(predictions, landmarks)

        # calculate the gradients
        loss_train_step.backward()

        # update the parameters
        optimizer.step()

        loss_train += loss_train_step.item()
        running_loss = loss_train/step

        print_overwrite(step, len(train_loader), running_loss, 'train')

    network.eval()
    with torch.no_grad():
        for step in range(1, len(valid_loader)+1):

            images, landmarls = next(iter(valid_loader))

            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0), -1).to(device)

            predictions = network(images)

            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step

            print_overwrite(step, len(valid_loader), running_loss, 'valid')
    
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)

    print('\n-------------------------------------------------------------')
    print(f'Epoch: {epoch}  Train Loss: {loss_train:.4f}  Valid Loss: {loss_valid:.4f}')
    print('-------------------------------------------------------------')

    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), 'Face Landmarks Detection/weights')
    


