import torch
import torch.nn as nn
import torch.optim as optim
# import torchmetrics
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import os
import time
from tqdm import tqdm
from pathlib import Path

from nn import Net
from dataset import datasets, imshow

# weights = 'classifier/models/19_25_34-02_04_2023/epoch_9_weights.pth'
weights = 'classifier/models/22_32_36-03_04_2023/epoch_9_weights.pth'

def checkTest(loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(loader):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy of the network on the {total} test images: {(correct / total):.2%}")

if __name__ == '__main__':
    trainloader, testloader = datasets()
    print(f"[INFO] Len of train loader: {len(trainloader)}")
    print(f"[INFO] Len of test loader: {len(testloader)}")

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    print(f"Labels: {labels}")

    net = Net()
    net.load_state_dict(torch.load(weights))
    
    checkTest(trainloader)
    checkTest(testloader)

    # outputs = net(images)

    # _, predicted = torch.max(outputs, 1)

    # # imshow(images, labels, predicted.numpy())

    # print(f"Predictions: {predicted}")
    # print(f"Predictions: {predicted.size()}")

    


