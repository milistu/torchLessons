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
from data import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Current running device: {device}")

def main(epoch_num:int=2):
    """
    Main function for model training and validation
    Input
    -----
    - epoch_num = number of training epochs, default 2
    """
    for epoch in tqdm(range(epoch_num)):
        train_loss =

if __name__ == '__min__':
    current_time = time.strftime('%H_%M_%S-%d_%m_%Y', time.localtime(time.time()))
    log_dir = Path(f'classifier/logs/{current_time}')

    batch_size = 4
    train_loader, test_loader = datasets(batch_size)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(log_dir)
    epoch_num = 10
    main(epoch_num)

    



