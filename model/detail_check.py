import numpy as np

def compute_features(landmarks):
    # 곡률 기반 특징 벡터 계산 (간단한 예시: 연속한 각도 차이 + 거리)
    diffs = np.diff(landmarks, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_diffs = np.diff(angles)
    distances = np.linalg.norm(diffs, axis=1)
    return np.concatenate([angle_diffs, distances])

def compute_similarity(f1, f2):
    # L2 거리 기반 유사도
    return np.linalg.norm(f1 - f2)

def align_landmarks(landmarks):
    # 중심 정렬 + 크기 정규화 (스케일/평행 이동 정렬)
    mean = np.mean(landmarks, axis=0)
    centered = landmarks - mean
    norm = np.linalg.norm(centered)
    if norm == 0:
        return centered
    return centered / norm
