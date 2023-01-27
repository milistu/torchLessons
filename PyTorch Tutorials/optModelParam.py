### OPTIMIZING MODEL PARAMETERS
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

## Prerequisite Code

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuraNetwork(nn.Module):
    def __init__(self):
        super(NeuraNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuraNetwork()

## Hyperparameters
'''
We define the following hyperparameters for training:
-----------------------------------------------------
    Number of Epochs - the number times to iterate over the dataset

    Batch Size - the number of data samples propagated through the network before the parameters are updated

    Learning Rate - how much to update models parameters at each batch/epoch. 
        Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
'''

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function


