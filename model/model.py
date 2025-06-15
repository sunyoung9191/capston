import torch.nn as nn
import torch

class MyLandmarkModel(nn.Module):
    def __init__(self):
        super(MyLandmarkModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 40),  # 🔥 20 points × (x, y)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)  # 🔥 정규화
        x = x.view(-1, 20, 2)  # 🔄 (batch_size, 40) → (batch_size, 20, 2)
        return x
