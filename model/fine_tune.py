# fine_tune.py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
from model_resnet import ResNetLandmarkModel
from torchvision import transforms

# 커스텀 데이터셋
class HardSampleDataset(Dataset):
    def __init__(self, df, image_dir, image_size=224):
        self.df = df
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        coords = row.iloc[0:44].values.astype('float32').reshape(-1, 2)
        coords = torch.tensor(coords / self.image_size, dtype=torch.float32).flatten()

        img_tensor = self.transform(img)
        return img_tensor, coords

# 하드 샘플 로드
df = pd.read_csv("pm0502bad_sample.csv")
image_dir = r"C:\Users\qkrgu\PycharmProjects\cap\venv\all"
dataset = HardSampleDataset(df, image_dir)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetLandmarkModel().to(device)
model.load_state_dict(torch.load("landmark_model.pth", map_location=device))

# 학습 구성
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.SmoothL1Loss()  # 또는 WingLoss 대체 가능

# 파인튜닝 루프 (1~2 epoch 추천)
model.train()
for epoch in range(2):
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "landmark_model_finetuned.pth")
