import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
import os
from torchvision import transforms
from model import MyLandmarkModel

# --------------------- 설정 ---------------------
csv_path = "bad_sample.csv"
image_dir = r"C:\Users\qkrgu\PycharmProjects\cap\venv\all"
model_path = "landmark_model_finetuned.pth"
output_dir = "psgred_vis"
os.makedirs(output_dir, exist_ok=True)

# --------------------- 모델 로드 ---------------------
model = MyLandmarkModel()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --------------------- 전처리 ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

# --------------------- RMSE 계산 함수 ---------------------
def compute_rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

# --------------------- 메인 루프 ---------------------
df = pd.read_csv(csv_path)

total_rmse = []
for idx, row in df.iterrows():
    img_path = os.path.join(image_dir, row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {img_path}")
        continue

    # 1. 원본 크기 저장
    h_orig, w_orig = img.shape[:2]

    # 2. 모델 입력용으로 resize
    input_img = cv2.resize(img, (224, 224))
    input_tensor = transform(input_img).unsqueeze(0)

    # 3. 예측
    with torch.no_grad():
        pred = model(input_tensor).squeeze().numpy().reshape(-1, 2)

    # 4. pred는 224x224 기준 → 원본 크기로 변환
    pred_denorm = pred * [w_orig / 224, h_orig / 224]

    # GT 좌표 22개 중에서 2개 제외 (예: 5, 6번째 인덱스 → Python index 기준)
    gt = []
    for i in range(22):
        gt.append([row[f'pt{i + 1}_x'], row[f'pt{i + 1}_y']])
    gt = np.array(gt)

    exclude_idx = [4, 5]  # 0-based index로 5번째와 6번째 제거
    include_idx = [i for i in range(22) if i not in exclude_idx]
    gt = gt[include_idx]

    h, w = img.shape[:2]
    gt_norm = gt / [w, h]  # 정규화
    pred_denorm = pred * [w, h]
    gt_denorm = gt_norm * [w, h]

    rmse = compute_rmse(pred_denorm, gt_denorm)
    total_rmse.append(rmse)

    # 🔍 시각화
    vis = img.copy()
    for i, ((gx, gy), (px, py)) in enumerate(zip(gt_denorm, pred_denorm)):
        cv2.circle(vis, (int(gx), int(gy)), 3, (0, 255, 0), -1)
        cv2.putText(vis, f"G{i + 1}", (int(gx) + 2, int(gy) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)
        cv2.putText(vis, f"P{i + 1}", (int(px) + 2, int(py) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

        cv2.line(vis, (int(gx), int(gy)), (int(px), int(py)), (255, 0, 255), 1)

    name = os.path.basename(img_path)
    cv2.putText(vis, f"RMSE: {rmse:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"debug_{name}"), vis)

# --------------------- 평균 RMSE 출력 ---------------------
avg_rmse = np.mean(total_rmse)
print(f"✅ 평균 RMSE: {avg_rmse:.2f}")
