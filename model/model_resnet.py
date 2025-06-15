# model_resnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=22):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_landmarks * 2)

    def forward(self, x):
        return self.backbone(x)
