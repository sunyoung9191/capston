import torch
import numpy as np
from tqdm import tqdm
import csv
from model import MyLandmarkModel
from dataset import EyeLandmarkDataset  # Dataset 클래스 경로에 맞게 수정
from torch.utils.data import DataLoader

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyLandmarkModel().to(device)
model.load_state_dict(torch.load("best_model.pth"))  # 모델 경로 맞게
model.eval()

# 예측 대상 전체 데이터셋 로드
dataset = EyeLandmarkDataset("data/labels.csv", "data/images", transform=your_transform, img_size=224)  # 경로 및 transform 맞게
dataloader = DataLoader(dataset, batch_size=1)

all_preds = []
filenames = []
large_error_list = []

for i, (img, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
    img = img.to(device)
    gt = gt.squeeze(0).numpy().reshape(-1, 2)

    with torch.no_grad():
        pred = model(img).cpu().squeeze(0).numpy()  # (20, 2)

    error = np.linalg.norm(pred - gt, axis=1)
    mean_error = np.mean(error)

    # 예측 저장
    all_preds.append(pred.flatten())
    filenames.append(dataset.data.iloc[i]['filename'])

    # 기준 이상 오차인 경우 기록
    if mean_error > 0.05:
        large_error_list.append([dataset.data.iloc[i]['filename'], *gt.flatten()])

# 전체 예측 저장
np.savez("predicted_landmarks_exclude_pupil.npz",
         landmarks=np.array(all_preds),
         filenames=np.array(filenames))

# 오차 큰 샘플 CSV 저장
with open("0528large_error_landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ['filename'] + [f'x{i}' for i in range(20)] + [f'y{i}' for i in range(20)]
    writer.writerow(header)
    for row in large_error_list:
        writer.writerow(row)
