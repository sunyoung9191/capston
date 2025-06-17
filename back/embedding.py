import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_similarity import EmbeddingNet

# 경로 설정
image_folder = "images"
csv_path = "data2.csv"
output_csv_path = "data_with_embedding.csv"

# 모델 세팅
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet()
model.load_state_dict(torch.load("eye_embedding_model.pth", map_location=device))
model = model.to(device).eval()

# 전처리
target_size = 224
transform = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 데이터 불러오기
df = pd.read_csv(csv_path)
df = df[df['image_before'].notna() & df['image_after'].notna()].copy()
df['image_before'] = df['image_before'].astype(str)
df['image_after'] = df['image_after'].astype(str)

# 임베딩 함수
def get_embedding(path):
    try:
        image = Image.open(path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor).squeeze().cpu().numpy()
        return emb.tolist()
    except Exception as e:
        print(f"❌ {path} 실패:", e)
        return [0.0] * 128

# before / after 둘 다 처리
df['embedding_before'] = df['image_before'].apply(lambda f: get_embedding(os.path.join(image_folder, f)))
df['embedding_after'] = df['image_after'].apply(lambda f: get_embedding(os.path.join(image_folder, f)))

# 저장
df.to_csv(output_csv_path, index=False)
print(f"✅ 저장 완료: {output_csv_path}")
