import torch
import torch.nn as nn
import torchvision.models as models

class MyLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=20):
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
            nn.Linear(512, num_landmarks * 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x.view(-1, -1, 2)  # (B, num_landmarks*2) -> (B, num_landmarks, 2)


class ResNetLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=22):
        super(ResNetLandmarkModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_landmarks * 2)

    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(x).view(-1, -1, 2)


def get_model(model_type='cnn', num_landmarks=20):
    if model_type == 'cnn':
        return MyLandmarkModel(num_landmarks)
    elif model_type == 'resnet':
        return ResNetLandmarkModel(num_landmarks)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
