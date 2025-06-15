import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """설정 클래스"""
    # 필수 설정 (GT가 있는 경우)
    csv_file: Optional[str] = None  # GT 비교용 (선택사항)
    csv_file = r'C:\Users\qkrgu\PycharmProjects\cap\venv\all\call_loc.csv'
    image_dir = r'C:\Users\qkrgu\PycharmProjects\cap\venv\all'
    model_path: str = 'landmark_model.pth'

    # 처리 설정
    image_size: int = 224
    mae_threshold: float = 2.0
    exclude_idx: List[int] = None
    include_idx: List[int] = None

    def __post_init__(self):
        if self.exclude_idx is None:
            self.exclude_idx = [4, 5]
        if self.include_idx is None:
            self.include_idx = [i for i in range(22) if i not in self.exclude_idx]


class LandmarkPredictor:
    """랜드마크 예측 클래스"""

    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """모델 로드"""
        try:
            from model import MyLandmarkModel
            model = MyLandmarkModel().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model
        except ImportError:
            logger.error("model.py에서 MyLandmarkModel을 import할 수 없습니다.")
            raise

    def predict_single(self, image_path: str, image_size: int = 224) -> Optional[np.ndarray]:
        """단일 이미지에서 랜드마크 예측"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.predict_from_array(img_rgb, image_size)
        except Exception as e:
            logger.warning(f"예측 실패 {image_path}: {e}")
            return None

    def predict_from_array(self, image: np.ndarray, image_size: int = 224) -> np.ndarray:
        """numpy 배열에서 랜드마크 예측"""
        # 이미지 전처리
        img_resized = cv2.resize(image, (image_size, image_size))
        img_tensor = torch.from_numpy(img_resized / 255.).permute(2, 0, 1)
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0).float().to(self.device)

        # 예측
        with torch.no_grad():
            pred = self.model(img_tensor).cpu().squeeze(0).numpy().reshape(-1, 2)

        return pred


class FeatureExtractor:
    """특징 추출 클래스"""

    @staticmethod
    def angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """세 점 사이의 각도 계산"""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    @staticmethod
    def curvature_score(pts: List[np.ndarray]) -> float:
        """곡률 점수 계산"""
        p1, p2, p3, p4 = pts
        angle1 = FeatureExtractor.angle_between(p1, p2, p3)
        angle2 = FeatureExtractor.angle_between(p2, p3, p4)
        return (angle1 + angle2) / 2

    @staticmethod
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """유클리드 거리 계산"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def extract_features(landmarks: np.ndarray) -> Dict[str, float]:
        """특징 추출 (랜드마크만으로)"""
        landmarks = np.array(landmarks).reshape(-1, 2)
        feats = {}

        # 기존 특징들 (원본 코드 기반)
        try:
            feats['front_curvature_L'] = FeatureExtractor.curvature_score([
                landmarks[2], landmarks[5], landmarks[6], landmarks[7]
            ])
            feats['front_angle_L'] = FeatureExtractor.angle_between(
                landmarks[2], landmarks[5], landmarks[6]
            )
            feats['tail_curvature_L'] = FeatureExtractor.curvature_score([
                landmarks[0], landmarks[4], landmarks[5], landmarks[6]
            ])
            feats['tail_angle_L'] = FeatureExtractor.angle_between(
                landmarks[0], landmarks[4], landmarks[5]
            )
            feats['front_curvature_R'] = FeatureExtractor.curvature_score([
                landmarks[3], landmarks[13], landmarks[14], landmarks[15]
            ])
            feats['front_angle_R'] = FeatureExtractor.angle_between(
                landmarks[3], landmarks[13], landmarks[14]
            )
            feats['tail_curvature_R'] = FeatureExtractor.curvature_score([
                landmarks[2], landmarks[12], landmarks[13], landmarks[14]
            ])
            feats['tail_angle_R'] = FeatureExtractor.angle_between(
                landmarks[2], landmarks[12], landmarks[13]
            )

            feats['eye_length_L'] = FeatureExtractor.euclidean_distance(landmarks[0], landmarks[6])
            feats['eye_length_R'] = FeatureExtractor.euclidean_distance(landmarks[3], landmarks[14])
            feats['asymmetry'] = abs(feats['eye_length_L'] - feats['eye_length_R'])

        except IndexError:
            logger.warning("랜드마크 인덱스 오류 - 기본값 사용")
            for key in ['front_curvature_L', 'front_angle_L', 'tail_curvature_L', 'tail_angle_L',
                        'front_curvature_R', 'front_angle_R', 'tail_curvature_R', 'tail_angle_R',
                        'eye_length_L', 'eye_length_R', 'asymmetry']:
                feats[key] = 0.0

        return feats


class SimilaritySearcher:
    """유사도 검색 클래스"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self.similarity_matrix = None
        self.fitted = False

    def fit(self, landmarks_list: List[np.ndarray]) -> None:
        """특징 행렬 구성 및 유사도 계산"""
        features = []
        for landmarks in landmarks_list:
            feat = FeatureExtractor.extract_features(landmarks)
            features.append(feat)

        if not features:
            logger.error("특징을 추출할 수 없습니다.")
            return

        feature_df = pd.DataFrame(features)

        # 가중치 적용
        weighted_df = feature_df.copy()
        important_features = ['front_curvature_L', 'front_angle_L',
                              'front_curvature_R', 'front_angle_R']
        for feat in important_features:
            if feat in weighted_df.columns:
                weighted_df[feat] *= 2.0

        # 정규화
        X_scaled = self.scaler.fit_transform(weighted_df.fillna(0))

        self.feature_matrix = X_scaled
        self.similarity_matrix = cosine_similarity(X_scaled)
        self.fitted = True

    def find_similar(self, query_idx: int, topk: int = 10) -> List[Tuple[int, float]]:
        """유사한 샘플 찾기"""
        if not self.fitted:
            raise ValueError("먼저 fit() 메서드를 호출해주세요.")

        similarities = self.similarity_matrix[query_idx]
        top_indices = np.argsort(similarities)[::-1][:topk + 1]

        return [(idx, similarities[idx]) for idx in top_indices]


class IntegratedLandmarkSystem:
    """통합 랜드마크 시스템 - 이미지만으로 작동"""

    def __init__(self, config: Config):
        self.config = config
        self.predictor = LandmarkPredictor(config.model_path)
        self.searcher = SimilaritySearcher()

        # 결과 저장
        self.image_files = []
        self.predictions = []
        self.ground_truths = []  # GT가 있는 경우에만
        self.maes = []  # GT가 있는 경우에만

    def process_images_from_directory(self, image_filter: str = None) -> None:
        """디렉토리에서 이미지들을 처리"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for filename in os.listdir(self.config.image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # 필터 적용 (예: 'after' 포함 이미지만)
                if image_filter and image_filter not in filename:
                    continue

                image_path = os.path.join(self.config.image_dir, filename)

                # 랜드마크 예측
                landmarks = self.predictor.predict_single(image_path, self.config.image_size)

                if landmarks is not None:
                    self.image_files.append(filename)
                    self.predictions.append(landmarks)
                    logger.info(f"✅ 처리 완료: {filename}")
                else:
                    logger.warning(f"❌ 처리 실패: {filename}")

        logger.info(f"총 {len(self.predictions)}개 이미지 처리 완료")

        # 유사도 검색 모델 학습
        if self.predictions:
            self.searcher.fit(self.predictions)
            logger.info("유사도 검색 모델 학습 완료")

    def process_with_ground_truth(self) -> None:
        """GT가 있는 경우 처리 (원본 평가 코드 기능)"""
        if not self.config.csv_file:
            logger.error("CSV 파일 경로가 설정되지 않았습니다.")
            return

        df = pd.read_csv(self.config.csv_file)

        for idx, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(self.config.image_dir, filename)

            if not os.path.exists(image_path):
                continue

            # 랜드마크 예측
            landmarks = self.predictor.predict_single(image_path, self.config.image_size)

            if landmarks is not None:
                # GT 좌표 추출
                coords = row.iloc[0:44].values.astype(np.float32).reshape(-1, 2)
                gt_landmarks = coords[self.config.include_idx] / self.config.image_size

                # MAE 계산
                mae = np.mean(np.abs(gt_landmarks - landmarks)) * self.config.image_size

                self.image_files.append(filename)
                self.predictions.append(landmarks)
                self.ground_truths.append(gt_landmarks)
                self.maes.append(mae)

                status = "❗ 오차 큼" if mae > self.config.mae_threshold else "✅ 정상"
                logger.info(f"{status} {filename} | MAE: {mae:.2f}")

        # 유사도 검색 모델 학습
        if self.predictions:
            self.searcher.fit(self.predictions)
            logger.info("유사도 검색 모델 학습 완료")

    def find_similar_images(self, query_idx: int, topk: int = 10) -> List[Tuple[int, float]]:
        """유사한 이미지 찾기"""
        return self.searcher.find_similar(query_idx, topk)

    def visualize_results(self, query_idx: int, topk: int = 5):
        """결과 시각화"""
        if query_idx >= len(self.predictions):
            logger.error(f"인덱스 {query_idx}가 범위를 벗어났습니다.")
            return

        similar_results = self.find_similar_images(query_idx, topk)

        fig, axes = plt.subplots(2, min(6, len(similar_results)), figsize=(20, 8))
        if len(similar_results) == 1:
            axes = axes.reshape(2, 1)

        for i, (idx, similarity) in enumerate(similar_results[:6]):
            # 원본 이미지
            image_path = os.path.join(self.config.image_dir, self.image_files[idx])
            img = Image.open(image_path)
            axes[0, i].imshow(img)
            title = f"Query" if i == 0 else f"Sim: {similarity:.3f}"
            axes[0, i].set_title(title)
            axes[0, i].axis('off')

            # 랜드마크 시각화
            img_resized = np.array(img.resize((self.config.image_size, self.config.image_size)))
            axes[1, i].imshow(img_resized)

            # 예측된 랜드마크
            pred = self.predictions[idx] * self.config.image_size
            axes[1, i].plot(pred[:, 0], pred[:, 1], 'rx', markersize=4, label='Pred')

            # GT가 있는 경우
            if self.ground_truths and idx < len(self.ground_truths):
                gt = self.ground_truths[idx] * self.config.image_size
                axes[1, i].plot(gt[:, 0], gt[:, 1], 'go', markersize=3, label='GT')
                mae_text = f"MAE: {self.maes[idx]:.2f}" if self.maes else ""
                axes[1, i].set_title(mae_text)
            else:
                axes[1, i].set_title("Predicted")

            axes[1, i].axis('off')
            axes[1, i].legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    def save_predictions_to_csv(self, output_path: str = "predicted_landmarks.csv"):
        """예측 결과를 CSV로 저장"""
        if not self.predictions:
            logger.error("저장할 예측 결과가 없습니다.")
            return

        data = []
        for i, (filename, landmarks) in enumerate(zip(self.image_files, self.predictions)):
            row = {'filename': filename}

            # 랜드마크 좌표 추가
            for j, (x, y) in enumerate(landmarks):
                row[f'pt{j + 1}_x'] = x * self.config.image_size
                row[f'pt{j + 1}_y'] = y * self.config.image_size

            # MAE 추가 (GT가 있는 경우)
            if self.maes and i < len(self.maes):
                row['mae'] = self.maes[i]

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"예측 결과가 {output_path}에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    config = Config()
    system = IntegratedLandmarkSystem(config)

    print("=== 통합 랜드마크 시스템 ===")
    print("1. 이미지만으로 랜드마크 예측 + 유사도 검색")
    print("2. GT와 비교하여 성능 평가 + 유사도 검색")

    mode = input("모드를 선택하세요 (1 또는 2): ").strip()

    if mode == "1":
        # 이미지만으로 처리
        filter_text = input("필터할 텍스트 (예: 'after', 없으면 엔터): ").strip()
        image_filter = filter_text if filter_text else None

        logger.info("이미지 처리 시작...")
        system.process_images_from_directory(image_filter)

        # 예측 결과 저장
        system.save_predictions_to_csv("predicted_landmarks.csv")

    elif mode == "2":
        # GT와 함께 처리
        csv_path = input(f"CSV 파일 경로 (기본값: {config.csv_file}): ").strip()
        if csv_path:
            config.csv_file = csv_path

        logger.info("GT와 함께 처리 시작...")
        system.process_with_ground_truth()

        # 좋은/나쁜 샘플 분류
        if system.maes:
            good_samples = [f for f, mae in zip(system.image_files, system.maes)
                            if mae <= config.mae_threshold]
            bad_samples = [f for f, mae in zip(system.image_files, system.maes)
                           if mae > config.mae_threshold]

            print(f"✅ 좋은 샘플: {len(good_samples)}개")
            print(f"❗ 나쁜 샘플: {len(bad_samples)}개")

    # 유사도 검색 데모
    if system.predictions:
        print(f"\n총 {len(system.predictions)}개 이미지로 유사도 검색 가능")

        # 랜덤 샘플 시각화
        demo_count = min(3, len(system.predictions))
        random_indices = random.sample(range(len(system.predictions)), demo_count)

        for idx in random_indices:
            print(f"\n🔍 Query: {system.image_files[idx]}")
            system.visualize_results(idx, topk=5)


if __name__ == "__main__":
    main()