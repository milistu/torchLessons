### Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

### Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(f"Initial check, expecting (64,10): {model(x).shape}")

### Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

### Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

### Load Data
train_dataset = datasets.MNIST(
    root = 'data',
    train = True,
    transform = transforms.ToTensor(), 
    download = True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(
    root = 'data',
    train = False,
    transform = transforms.ToTensor(), 
    download = True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

### Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

### Loss and optimizer
criterion = nn.CrossEntropyLoss()
optmizer = optim.Adam(model.parameters(), lr=learning_rate)

### Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        print(data.shape)
        break
    break