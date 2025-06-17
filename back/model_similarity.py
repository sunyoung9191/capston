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
IMAGE_FOLDER = 'images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리 설정
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class TripletDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png')) and len(f.split('_')) > 2]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        anchor_file = self.image_files[idx]
        anchor_path = os.path.join(self.image_dir, anchor_file)
        anchor_img = Image.open(anchor_path).convert("RGB")

        try:
            anchor_key = anchor_file.split('_')[1]
        except IndexError:
            anchor_key = "UNKNOWN"

        # 포지티브 후보: 같은 그룹, 다른 파일
        positive_candidates = [f for f in self.image_files
                               if f != anchor_file and len(f.split('_')) > 2 and f.split('_')[1] == anchor_key]
        if not positive_candidates:
            positive_file = anchor_file
        else:
            positive_file = random.choice(positive_candidates)
        positive_img = Image.open(os.path.join(self.image_dir, positive_file)).convert("RGB")

        # 네거티브 후보: 다른 그룹
        negative_candidates = [f for f in self.image_files
                               if len(f.split('_')) > 2 and f.split('_')[1] != anchor_key]
        negative_file = random.choice(negative_candidates)
        negative_img = Image.open(os.path.join(self.image_dir, negative_file)).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


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

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
