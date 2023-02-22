import torch
from torch.utils.data import DataLoader, Dataset, random_split

import data

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

