import os
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def datasets(batch_size:int = 4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = transform,
    )

    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)

    testset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = transform,
    )

    testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)

    return trainloader, testloader

# HELPERS:

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    np_img = img.numpy()

    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    print(f"[INFO] Current working directory: {os.getcwd()}")

    batch_size = 4
    trainloader, testloader = datasets(batch_size)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))
