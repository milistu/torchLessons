import os
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


if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    
    trainloader, testloader = datasets()
