import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torchvision import models

try:
    os.remove("features.npy")
    print("✅ features.npy 삭제 완료")
except FileNotFoundError:
    print("⚠️ features.npy 파일이 없음")

try:
    os.remove("filenames.pkl")
    print("✅ filenames.pkl 삭제 완료")
except FileNotFoundError:
    print("⚠️ filenames.pkl 파일이 없음")

# ===== 모델 정의 (ResNet18 기반) =====
class EyeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        # ResNet18 순서대로 feature 추출
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        features = x
        logits = self.model.fc(x)
        return features if return_features else logits

# ===== 설정 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path      = "eye_classifier.pth"
image_dir       = r"C:\Users\parks\PycharmProjects\capston\venv\cropped_eyes"
test_image_path = r"C:\Users\parks\PycharmProjects\capston\venv\test_image2.jpg"
top_k           = 5

# ===== 전처리 정의 =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===== 모델 불러오기 =====
model = EyeClassifier(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== 학습 이미지 벡터 로딩 (이미 저장된 경우) =====
if os.path.exists("features.npy") and os.path.exists("filenames.pkl"):
    features = np.load("features.npy")
    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)
else:
    # 저장이 안 되어 있으면 한 번 생성 후 저장
    features = []
    filenames = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith("_before.jpg"):
            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor, return_features=True).cpu().numpy().flatten()
            features.append(feat)
            filenames.append(fname)
    features = np.array(features)
    np.save("features.npy", features)
    with open("filenames.pkl", "wb") as f:
        pickle.dump(filenames, f)
print(f"✅ Loaded {len(filenames)} feature vectors.")

# ===== 테스트 이미지 임베딩 추출 =====
test_img = Image.open(test_image_path).convert('RGB')
test_tensor = transform(test_img).unsqueeze(0).to(device)
with torch.no_grad():
    test_feat = model(test_tensor, return_features=True).cpu().numpy()

# ===== 유사도 계산 =====
sims = cosine_similarity(test_feat, features)[0]
top_indices = np.argsort(sims)[-top_k:][::-1]

# ===== 시각화 (3행 × 5열) =====
fig, axes = plt.subplots(3, top_k, figsize=(15, 9))
fig.suptitle("Test + Top 5 Similar Eyes", fontsize=18)

# 1행: 테스트 이미지 (0열만 사용)
axes[0, 0].imshow(np.array(test_img))
axes[0, 0].set_title("Test Image")
axes[0, 0].axis('off')
for col in range(1, top_k):
    axes[0, col].axis('off')

# 2행: Before 이미지
for i, idx in enumerate(top_indices):
    before_file = filenames[idx]
    before_path = os.path.join(image_dir, before_file)
    before_img  = Image.open(before_path).convert('RGB')
    axes[1, i].imshow(np.array(before_img))
    axes[1, i].set_title(f"Before #{i+1}")
    axes[1, i].axis('off')

# 3행: After 이미지
for i, idx in enumerate(top_indices):
    before_file = filenames[idx]
    after_file  = before_file.replace("_before.jpg", "_after.jpg")
    after_path  = os.path.join(image_dir, after_file)
    if os.path.exists(after_path):
        after_img = Image.open(after_path).convert('RGB')
        axes[2, i].imshow(np.array(after_img))
    else:
        axes[2, i].text(0.5, 0.5, "No After", ha='center', va='center')
        axes[2, i].set_facecolor('lightgray')
    axes[2, i].set_title(f"After #{i+1}")
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()
