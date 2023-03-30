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

from nn import Net
from data import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Current running device: {device}")
