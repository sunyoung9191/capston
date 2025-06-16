import cv2
import os
import numpy as np

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

input_folder = "C:/Users/qkrgu/Downloads/Mind_lee2/Mind_lee"
output_folder = "cropped_eyes"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def resize_and_pad(image, size=(224, 224)):
    """이미지를 224x224로 리사이징 & 패딩"""
    h, w, _ = image.shape
    target_h, target_w = size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 패딩 추가
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded_image = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image


def detect_and_crop_eyes(image_path, save_path):
    """눈을 검출하여 크롭 후 224x224로 변환하여 저장"""
    image = cv2.imread(image_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) == 0:
        return False  # 눈 검출 실패

    # 여러 개의 눈이 검출될 경우, 가장 큰 눈 2개 선택 (양쪽 눈)
    eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]  # 면적 기준 정렬

    # 눈의 좌표를 기반으로 전체 눈 영역 설정
    x_min = min([eye[0] for eye in eyes])
    y_min = min([eye[1] for eye in eyes])
    x_max = max([eye[0] + eye[2] for eye in eyes])
    y_max = max([eye[1] + eye[3] for eye in eyes])

    y_min += 10  # 윗부분을 잘라서 눈썹 제거

    # 크롭
    cropped_eye = image[y_min:y_max, x_min:x_max]

    if cropped_eye.size > 0:
        processed_eye = resize_and_pad(cropped_eye, size=(224, 224))
        cv2.imwrite(save_path, processed_eye)
        return True
    return False

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        output_path = os.path.join(output_folder, filename)

        if detect_and_crop_eyes(input_path, output_path):
            print(f"{filename} -> 눈 크롭 & 224x224 전처리 완료!")
        else:
            print(f"{filename} -> 눈 검출 실패!")

print("모든 이미지 처리 완료!")