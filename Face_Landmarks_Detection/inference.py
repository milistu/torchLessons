import time
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import network as net
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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎰 Device used: {device}")

start_time = time.time()

with torch.no_grad():

    best_network = net.Network()
    best_network.to(device)
    best_network.load_state_dict(torch.load('Face_Landmarks_Detection/weights/face_landmarks.pth'))
    best_network.eval()

    images, landmarks = next(iter(valid_loader))

    images = images.to(device)
    landmarks = (landmarks + 0.5) * 244

    predictions = best_network(images)
    predictions = (predictions.view(-1, 68, 2).cpu().numpy() + 0.5) * 224

    plt.figure(figsize=(10, 10))

    for img_num in range(8):
        plt.subplot(2, 4, img_num+1)
        plt.imshow(images[img_num].cpu().numpy().squeeze(), cmap='gray')
        plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c = 'r', s = 5)
        plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c = 'g', s = 5)

print(f"Total number of test images: {len(valid_dataset)}")
print(f"Elapsed Time: {time.time() - start_time}")
plt.suptitle("Inference results")
plt.show()
