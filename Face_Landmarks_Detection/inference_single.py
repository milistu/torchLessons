import time
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import network as net
import data 

# Create the Dataset
dataset = data.FaceLandmarksDataset(data.Transforms())

image, landmarks = dataset[11]
landmarks = (landmarks + 0.5) * 224
# image = image.unsqueeze(0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎰 Device used: {device}")

start_time = time.time()

with torch.no_grad():

    best_network = net.Network()
    best_network.to(device)
    best_network.load_state_dict(torch.load('Face_Landmarks_Detection/weights/face_landmarks.pth'))
    best_network.eval()

    image = image.to(device)
    # landmarks = (landmarks + 0.5) * 244

    predictions = best_network(image.unsqueeze(0))
    predictions = (predictions.view(68,2).cpu().numpy() + 0.5) * 224

    plt.figure(figsize=(10, 10))
    plt.imshow(image.cpu().numpy().squeeze(), cmap='gray')
    plt.scatter(predictions[:, 0], predictions[:, 1], c = 'r', s = 5)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c = 'g', s = 8)

# print(f"Total number of test images: {len(valid_dataset)}")
print(f"Elapsed Time: {time.time() - start_time}")
plt.title("Inference results")
plt.show()
