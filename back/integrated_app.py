from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import torch.nn as nn
# ===== 예은이의 커스텀 모델 구조 =====
from model_similarity import EmbeddingNet
from model import MyLandmarkModel
from view_pic_feature import FeatureExtractor

class MyLandmarkModel(nn.Module):
    def __init__(self):
        super(MyLandmarkModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 기존 저장된 모델 구조
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
    nn.ReLU(),                     # classifier.0
    nn.Linear(64 * 28 * 28, 512),  # classifier.1
    nn.ReLU(),                     # classifier.2
    nn.Linear(512, 44)             # classifier.3
)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ===== Flask 세팅 =====
app = Flask(__name__)
CORS(app)

# ===== 디바이스 설정 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 전처리 설정 =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===== 모델 로드 =====
embedding_model = EmbeddingNet().to(device)
embedding_model.load_state_dict(torch.load("eye_embedding_model.pth", map_location=device))
embedding_model.eval()


landmark_model = MyLandmarkModel().to(device)
landmark_model.load_state_dict(torch.load("landmark_model.pth", map_location=device))
landmark_model.eval()


# ===== CSV 데이터 로드 =====
df = pd.read_csv("data_final.csv")
df['embedding'] = df['embedding'].apply(ast.literal_eval)
df['landmarks'] = df['landmarks'].apply(ast.literal_eval)

# 랜드마크 기반 feature 추출
def extract_feat_from_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 2)
    return list(FeatureExtractor.extract_features(coords).values())

df['landmark_feat'] = df['landmarks'].apply(extract_feat_from_landmarks)

# ===== 유사도 계산 함수 =====
def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def get_embedding(img_tensor):
    with torch.no_grad():
        feat = embedding_model(img_tensor.unsqueeze(0).to(device))
    return feat.squeeze().cpu().numpy()

def get_landmark_feat(img_tensor):
    with torch.no_grad():
        pred = landmark_model(img_tensor.unsqueeze(0).to(device))
    coords = pred.view(-1, 2).cpu().numpy()
    return list(FeatureExtractor.extract_features(coords).values())

import html
import re
def clean_text(text):
    if isinstance(text, str):
        text = html.unescape(text)  # HTML 엔티티 제거
        text = re.sub(r'[\n\r\t]', ' ', text)  # 줄바꿈 제거
        text = re.sub(r'["\'\\]', '', text)    # 큰따옴표, 작은따옴표, 백슬래시 제거
        return text.strip()
    return ''


# ===== 이미지 서빙 엔드포인트 (Android에서 접근할 때 필요) =====
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# ===== 예측 API =====
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '이미지를 첨부해주세요'}), 400

    try:
        image = Image.open(request.files['image']).convert('RGB')
        img_tensor = transform(image)

        user_emb = get_embedding(img_tensor)
        user_land_feat = get_landmark_feat(img_tensor)

        results = []
        for _, row in df.iterrows():
            emb_sim = cosine_sim(user_emb, row['embedding'])
            land_sim = cosine_sim(user_land_feat, row['landmark_feat'])
            total_sim = 0 * emb_sim + 1.0 * land_sim


            results.append({
                "before": row['image_before'],  # 예: "b130.jpg"
                "after": row['image_after'],  
                "hospital": row['clinic_name'],
                "doctor": row['doctor_name'],
                "procedure": clean_text(row["procedure_type"]),
                "similarity": round(total_sim, 4)
            })

        # 병원 중복 없이 Top 5 추출
        seen = set()
        top5 = []
        for item in sorted(results, key=lambda x: x['similarity'], reverse=True):
            if item['hospital'] not in seen:
                top5.append(item)
                seen.add(item['hospital'])
            if len(top5) == 10:
                break

        return jsonify(top5)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ===== 서버 실행 =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
