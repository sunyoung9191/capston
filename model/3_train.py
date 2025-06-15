import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from model import MyLandmarkModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

# ====== 개선된 EyeLandmarkDataset (Albumentations 적용) ======
class EyeLandmarkDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, image_size=224):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size

        self.data = self.data[self.data['filename'].apply(
            lambda x: os.path.exists(os.path.join(self.image_dir, x))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        landmarks = row[:44].values.astype(np.float32).reshape(-1, 2)
        exclude_idx = [4, 5]
        landmarks = np.delete(landmarks, exclude_idx, axis=0)
        landmarks = landmarks / self.image_size

        if self.transform:
            augmented = self.transform(image=image, keypoints=landmarks)
            image = augmented['image']
            landmarks = np.array(augmented['keypoints'])
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

        image = (image - 0.5) / 0.5
        landmarks = torch.tensor(landmarks, dtype=torch.float32).flatten()

        return image, landmarks

# ====== 학습 하이퍼파라미터 ======
image_dir = r"C:\Users\qkrgu\PycharmProjects\cap\venv\crop_pop393"
csv_file = r"C:\Users\qkrgu\PycharmProjects\cap\venv\crop_pop393\0523_pop393_landmarks.csv"
batch_size = 32
num_epochs = 50
lr = 1e-4
model_save_path = "landmark_model_aug.pth"

# ====== Augmentation 정의 ======
train_transform = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=10, p=0.5),
    RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], keypoint_params={"format": "xy", "remove_invisible": False})

val_transform = Compose([
    ToTensorV2()
], keypoint_params={"format": "xy", "remove_invisible": False})

# ====== 데이터셋 로딩 및 분할 ======
full_dataset = EyeLandmarkDataset(csv_file, image_dir, transform=None)
train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)

train_dataset = EyeLandmarkDataset(csv_file, image_dir, transform=train_transform)
val_dataset = EyeLandmarkDataset(csv_file, image_dir, transform=val_transform)

train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False)

# ====== 모델 및 최적화 세팅 ======
model = MyLandmarkModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

if os.path.exists(model_save_path):
    print("📂 이전 모델을 불러옵니다.")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

# ====== 학습 루프 ======
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        targets = targets.view(-1, 20, 2)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    self.output = nn.Sequential(
        nn.Linear(..., 40),
        nn.Sigmoid()  # 출력값을 0~1 사이로
    )

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            targets = targets.view(-1, 20, 2)

            preds = model(imgs)
            loss = criterion(preds, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"[Epoch {epoch+1}/{num_epochs}] 🔹Train Loss: {avg_loss:.4f} | 🔸Val Loss: {val_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")

# 최종 모델 저장
torch.save(model.state_dict(), model_save_path)
print("✅ 모델 학습 완료 및 저장됨.")

# ====== 시각화 ======
model.eval()
with torch.no_grad():
    sample_dataset = EyeLandmarkDataset(csv_file, image_dir, transform=val_transform)
    img, gt = sample_dataset[0]
    pred = model(img.unsqueeze(0).to(device)).cpu().squeeze(0).numpy().reshape(-1, 2)
    gt = gt.numpy().reshape(-1, 2)

    img_np = img.permute(1, 2, 0).numpy() * 0.5 + 0.5

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.scatter(gt[:, 0] * 224, gt[:, 1] * 224, c='g', label='GT', s=20)
    plt.scatter(pred[:, 0] * 224, pred[:, 1] * 224, c='r', label='Pred', s=20)
    plt.legend()
    plt.title("예측 vs 실제 랜드마크")
    plt.axis("off")
    plt.show()
