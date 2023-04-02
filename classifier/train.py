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

def setDevice():
    """
    Set-up cuda device if available
    Output
    ------
    - device = device used by PC for training and vlaidation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Current running device: {device}")
    return device

def train(epoch:int):
    model.train()
    running_loss = 0.
    train_loss = 0.

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[INFO] [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            writer.add_scalar('Train Loss [per 2000]', running_loss / 2000, epoch * len(train_loader) + i)
            running_loss = 0.

    train_loss /= len(train_loader)
    return train_loss

def test(epoch:int):
    model.eval()
    running_loss = 0.
    test_loss = 0.

    with torch.no_grad():
        for i, (inputs, labels), in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[INFO] [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                writer.add_scalar('Test Loss [per 2000]', running_loss / 2000, epoch * len(test_loader) + i)
                running_loss = 0.
        
        test_loss /= len(test_loader)
    
    return test_loss

def main(epoch_num:int=2):
    """
    Main function for model training and validation
    Input
    -----
    - epoch_num = number of training epochs, default 2
    """
    if not os.path.exists(f'classifier/models/{current_time}'):
        os.makedirs(f'models/{current_time}')
    for epoch in tqdm(range(epoch_num)):
        train_loss =train(epoch)
        test_loss = test(epoch)
        print(f'[INFO] Epoch {epoch+1}/{epoch_num}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        writer.add_scalar('Train Loss [epoch]', train_loss, epoch)
        writer.add_scalar('Test Loss [epoch]', test_loss, epoch)
        
        torch.save(model.state_dict(), f'classifier/models/{current_time}/epoch_{epoch}_weights.pth')

    print("[INFO] Finished traning!")

if __name__ == '__main__':
    device = setDevice()
    current_time = time.strftime('%H_%M_%S-%d_%m_%Y', time.localtime(time.time()))
    log_dir = Path(f'classifier/logs/{current_time}')

    batch_size = 4
    train_loader, test_loader = datasets(batch_size)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(log_dir)
    epoch_num = 5
    main(epoch_num)

    



