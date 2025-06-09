import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 설정
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
EMBEDDING_SIZE = 128
IMAGE_FOLDER = 'cropped_eyes'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리 설정
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # RGB 기준
])

class TripletDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        anchor_path = os.path.join(self.image_dir, self.image_files[idx])
        anchor_img = Image.open(anchor_path).convert("RGB")

        positive_img = anchor_img.copy()

        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.image_files) - 1)
        negative_path = os.path.join(self.image_dir, self.image_files[neg_idx])
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# CNN 임베딩 모델
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), embedding_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.cnn(x))

# Triplet Loss 함수
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = F.pairwise_distance(anchor, positive)
    d_an = F.pairwise_distance(anchor, negative)
    loss = F.relu(d_ap - d_an + margin)
    return loss.mean()

def train():
    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE).to(device)

    if os.path.exists("eye_embedding_model.pth"):
        model.load_state_dict(torch.load("eye_embedding_model.pth", map_location=device))
        print("기존 모델 로드 완료")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TripletDataset(IMAGE_FOLDER, transform=transform)
    print(f"전체 학습 이미지 수: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = triplet_loss(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "eye_embedding_model.pth")
    print("모델 저장 완료: eye_embedding_model.pth")


if __name__ == "__main__":
    train()
