#모델의 랜드마크 예측과 GT값과의 비교 후 MAE 기반으로 좋은 샘플/나쁜 샘플을 분류하여 결과를 시각화 + 파일 저장 + 좌표 저장
import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import MyLandmarkModel

# 설정
csv_file = r'C:\Users\qkrgu\PycharmProjects\cap\venv\all\call_loc.csv'
image_dir = r'C:\Users\qkrgu\PycharmProjects\cap\venv\all'
model_path = 'landmark_model.pth'
image_size = 224
mae_threshold = 2

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyLandmarkModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 좌표 index 설정
exclude_idx = [4, 5]
include_idx = [i for i in range(22) if i not in exclude_idx]

# 결과 저장용 리스트
bad_samples = []
good_samples = []
bad_imgs = []

# 시각화 함수 (2초간 보여주고 자동 종료)
def plot_and_save_landmarks(img, gt_landmarks, pred_landmarks, save_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)

    gt = gt_landmarks * image_size
    pred = pred_landmarks * image_size

    plt.plot(gt[:, 0], gt[:, 1], 'go', label='GT (Green)')
    plt.plot(pred[:, 0], pred[:, 1], 'bx', label='Pred (Blue)')

    plt.legend(loc='upper right', fontsize=8)
    plt.axis('off')
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=150)

    # 자동 표시 후 닫기
    #plt.pause(2)
    plt.close()

# 데이터프레임 로드
df = pd.read_csv(csv_file)

# 메인 루프
for idx, row in df.iterrows():
    filename = row['filename']
    img_path = os.path.join(image_dir, filename)

    #print(f"🧪 시도 중: {img_path}")
    if not os.path.exists(img_path):
        #print(f"❌ 파일 존재 안함: {filename}")
        bad_imgs.append(filename)
        continue

    img = cv2.imread(img_path)
    if img is None:
        #print(f"❌ OpenCV 로드 실패: {filename}")
        bad_imgs.append(filename)
        continue

    # GT 좌표 정제
    coords = row.iloc[0:44].values.astype(np.float32).reshape(-1, 2)
    gt_landmarks = coords[include_idx] / image_size

    # 이미지 전처리
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_tensor = torch.from_numpy(img_resized / 255.).permute(2, 0, 1)
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).float().to(device)

    # 예측
    with torch.no_grad():
        pred = model(img_tensor).cpu().squeeze(0).numpy().reshape(-1, 2)
        pred_landmarks = pred

    mae = np.mean(np.abs(gt_landmarks - pred_landmarks)) * image_size

    if mae > mae_threshold:
        print(f"❗ {filename} | MAE: {mae:.2f} → 오차 큼 (저장 + 표시)")
        bad_samples.append(row)

        os.makedirs("bad_plots", exist_ok=True)
        save_path = os.path.join("bad_plots", f"{idx}_{filename}.png")
    else:
        print(f"✅ {filename} | MAE: {mae:.2f} → 정상 샘플 (저장 + 표시)")
        good_samples.append(row)

        os.makedirs("good_plots", exist_ok=True)
        save_path = os.path.join("good_plots", f"{idx}_{filename}.png")

    # 바로 보여주고 저장
    plot_and_save_landmarks(img_resized, gt_landmarks, pred_landmarks, save_path)

# 결과 저장
pd.DataFrame(bad_samples).to_csv("pm0511bad_sample.csv", index=False)
pd.DataFrame(good_samples).to_csv("pm0511good_sample.csv", index=False)

# 누락 이미지 출력
print("🚫 누락된 이미지 목록:")
print(bad_imgs)
