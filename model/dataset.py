import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EyeLandmarkDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size=224, transform=True):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.img_size = img_size
        self.transform = transform

        # 이미지 파일 존재하는 것만 필터링
        self.data = self.data[self.data['filename'].apply(
            lambda x: os.path.exists(os.path.join(self.image_dir, x))
        )].reset_index(drop=True)

        # Albumentations 증강 파이프라인 정의
        if self.transform:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=10, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.4
                ),
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.aug = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = cv2.imread(img_path)

        if image is None:
            # 이미지 읽기 실패 시 다음 인덱스로 대체
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 전체 22개의 좌표 중 2개 제외 (5, 6번째 → 인덱스 4, 5)
        full_landmarks = row[:44].values.astype(np.float32).reshape(-1, 2)
        exclude_indices = [4, 5]
        filtered = np.delete(full_landmarks, exclude_indices, axis=0)  # (20, 2)

        # 증강 적용
        aug = self.aug(image=image, keypoints=filtered)
        image = aug['image']                      # (3, H, W) tensor
        landmarks = np.array(aug['keypoints'])    # (20, 2)

        # 0~1 정규화
        landmarks = landmarks / self.img_size
        landmarks = torch.tensor(landmarks.flatten(), dtype=torch.float32)

        return image, landmarks
