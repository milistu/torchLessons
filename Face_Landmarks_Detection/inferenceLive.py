import time
import cv2
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

import data
import network as net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎰 Device used: {device}")


best_model = net.Network()
best_model.load_state_dict(torch.load('Face_Landmarks_Detection/weights/face_landmarks.pth', map_location=torch.device('cpu')))
# best_model.to(device)

faces_cascade = cv2.CascadeClassifier('Face_Landmarks_Detection/weights/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while(True):
    if cap.isOpened():
        ret, frame = cap.read()
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        faces = faces_cascade.detectMultiScale(grayscale_image, 1.1, 4)

        all_landmarks = []
        for (x, y, w, h) in faces:
            image = grayscale_image[y:y+h, x:x+w]
            image = TF.resize(Image.fromarray(image), size=(224, 224))
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            # image.to(device)

            with torch.no_grad():
                landmarks = best_model(image.unsqueeze(0))
            
            landmarks = (landmarks.view(68,2).cpu().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
            all_landmarks.append(landmarks)
        
        for landmarks in all_landmarks:
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), radius=2, color=(150, 255, 0), thickness=-1)

    
        cv2.imshow('Face Landmarks Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
