import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import pandas as pd
import os

# ====== 설정 ======
root_dir = r"C:\Users\qkrgu\PycharmProjects\cap\venv\cropped_amond"

# ====== 전역 변수 ======
landmarks = []
img = None
img_copy = None
img_path = None
results = []
current_guides = []
img_with_guides = None

# ====== 스냅 함수 ======
def snap_to_nearest_guide(x, y, guides):
    if not guides:
        return x, y
    closest = min(guides, key=lambda gx: abs(x - gx))
    return int(closest), int(y)

# ====== 클릭 이벤트 ======
def click_event(event, x, y, flags, param):
    global landmarks, img, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(landmarks) < 6:
            landmarks.append((x, y))
            redraw_landmarks()

            if len(landmarks) == 6:
                draw_guidelines()
                print("➡️ 기준선 위/아래 점을 각각 클릭하세요 (왼쪽 눈 8점, 오른쪽 눈 8점)")

        elif 6 <= len(landmarks) < 22:
            x, y = snap_to_nearest_guide(x, y, current_guides)
            landmarks.append((x, y))
            redraw_landmarks()

        if len(landmarks) == 22:
            cv2.destroyAllWindows()
            draw_final_curves_and_save()


def classify_pupils_and_draw(landmarks, left_eye_idx, right_eye_idx):
    pupil1, pupil2 = landmarks[4], landmarks[5]
    pupil_center1 = np.array(pupil1)
    pupil_center2 = np.array(pupil2)

    # 왼쪽 눈, 오른쪽 눈 중심 계산
    left_center = np.mean([landmarks[i] for i in left_eye_idx], axis=0)
    right_center = np.mean([landmarks[i] for i in right_eye_idx], axis=0)

    # 어느 동공이 어느 눈에 속하는지 자동 판별
    if np.linalg.norm(pupil_center1 - left_center) < np.linalg.norm(pupil_center1 - right_center):
        left_pupil = pupil1
        right_pupil = pupil2
    else:
        left_pupil = pupil2
        right_pupil = pupil1

    return left_pupil, right_pupil


# ====== 기준선 그리기 ======
def draw_guidelines():
    global landmarks, img, current_guides, img_with_guides

    lt_tail, lt_inner = landmarks[0], landmarks[1]
    rt_inner, rt_tail = landmarks[2], landmarks[3]
    pupil_left, pupil_right = landmarks[4], landmarks[5]

    current_guides = []

    def compute_guides(pt1, pt2):
        guides = []
        line = np.array(pt2) - np.array(pt1)
        for t in np.linspace(0, 1, 6)[1:-1]:  # 중간 4등분
            p = np.array(pt1) + t * line
            guides.append(p[0])
            cv2.line(img, (int(p[0]), 0), (int(p[0]), img.shape[0]), (255, 0, 255), 1)
        return guides

    current_guides += compute_guides(lt_tail, lt_inner)
    current_guides += compute_guides(rt_inner, rt_tail)

    eye_center_y = int((lt_tail[1] + lt_inner[1] + rt_inner[1] + rt_tail[1]) / 4)
    cv2.line(img, (0, eye_center_y), (img.shape[1], eye_center_y), (0, 255, 255), 1)

    pupil_center = ((pupil_left[0] + pupil_right[0]) // 2, (pupil_left[1] + pupil_right[1]) // 2)
    pupil_half_height = abs(pupil_center[1] - eye_center_y)
    cv2.line(img, (pupil_center[0], eye_center_y - pupil_half_height),
             (pupil_center[0], eye_center_y + pupil_half_height), (0, 200, 200), 2)

    img_with_guides = img.copy()
    cv2.imshow("Click landmarks", img)

# ====== 리렌더 ======
def redraw_landmarks():
    global img, img_with_guides, landmarks
    img = img_with_guides.copy() if img_with_guides is not None else img_copy.copy()
    for px, py in landmarks:
        cv2.circle(img, (px, py), 3, (0, 255, 0), -1)
    cv2.imshow("Click landmarks", img)

# ====== 곡선 그리기 ======
def draw_curve(points, color='r', smoothness=0.5):
    if len(points) < 4: return
    points = np.array(points)
    tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothness)
    unew = np.linspace(0, 1, 200)
    out = splev(unew, tck)
    plt.plot(out[0], out[1], color=color, lw=2)

# ====== 최종 출력 ======
def draw_final_curves_and_save():
    global landmarks, img_copy, img_path, results

    pts = np.array(landmarks)
    lt_tail, lt_inner = landmarks[0], landmarks[1]
    rt_inner, rt_tail = landmarks[2], landmarks[3]
    if landmarks[4][0] < landmarks[5][0]:
        pupil_left, pupil_right = landmarks[4], landmarks[5]
    else:
        pupil_left, pupil_right = landmarks[5], landmarks[4]

    # 눈 중심
    left_eye_center = (
        (lt_tail[0] + lt_inner[0]) / 2,
        (lt_tail[1] + lt_inner[1]) / 2
    )
    right_eye_center = (
        (rt_inner[0] + rt_tail[0]) / 2,
        (rt_inner[1] + rt_tail[1]) / 2
    )

    # 눈 벡터 및 단위 벡터
    eye_vec_left = np.array(lt_inner) - np.array(lt_tail)
    eye_unit_left = eye_vec_left / np.linalg.norm(eye_vec_left)

    eye_vec_right = np.array(rt_tail) - np.array(rt_inner)
    eye_unit_right = eye_vec_right / np.linalg.norm(eye_vec_right)

    # 동공 길이 (원래 입력된 좌우 길이)
    pupil_vec = np.array(pupil_right) - np.array(pupil_left)
    pupil_length = np.linalg.norm(pupil_vec)

    # 정면 보정된 동공 좌우점 - 각 눈
    half_proj = pupil_length / 2

    adjusted_pupil_left_start = (
        left_eye_center[0] - half_proj * eye_unit_left[0],
        left_eye_center[1] - half_proj * eye_unit_left[1]
    )
    adjusted_pupil_left_end = (
        left_eye_center[0] + half_proj * eye_unit_left[0],
        left_eye_center[1] + half_proj * eye_unit_left[1]
    )

    adjusted_pupil_right_start = (
        right_eye_center[0] - half_proj * eye_unit_right[0],
        right_eye_center[1] - half_proj * eye_unit_right[1]
    )
    adjusted_pupil_right_end = (
        right_eye_center[0] + half_proj * eye_unit_right[0],
        right_eye_center[1] + half_proj * eye_unit_right[1]
    )

    # 곡선 좌표 분리
    left_upper = [lt_tail] + landmarks[6:9] + [lt_inner]
    left_lower = [lt_tail] + landmarks[10:13] + [lt_inner]
    right_upper = [rt_inner] + landmarks[14:17] + [rt_tail]
    right_lower = [rt_inner] + landmarks[18:21] + [rt_tail]

    # 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.scatter(pts[:, 0], pts[:, 1], c='blue', s=10)

    draw_curve(left_upper, 'red')
    draw_curve(left_lower, 'red')
    draw_curve(right_upper, 'blue')
    draw_curve(right_lower, 'blue')

    # 보정된 동공 선분 시각화
    plt.plot(
        [adjusted_pupil_left_start[0], adjusted_pupil_left_end[0]],
        [adjusted_pupil_left_start[1], adjusted_pupil_left_end[1]],
        'c-', lw=2, label='왼쪽 보정 동공 길이'
    )
    plt.plot(
        [adjusted_pupil_right_start[0], adjusted_pupil_right_end[0]],
        [adjusted_pupil_right_start[1], adjusted_pupil_right_end[1]],
        'm-', lw=2, label='오른쪽 보정 동공 길이'
    )
    plt.scatter([left_eye_center[0]], [left_eye_center[1]], color='cyan', marker='x', s=40, label='왼눈 중심')
    plt.scatter([right_eye_center[0]], [right_eye_center[1]], color='magenta', marker='x', s=40, label='오른눈 중심')

    # pupil height
    eye_line_y = (lt_tail[1] + lt_inner[1] + rt_inner[1] + rt_tail[1]) / 4
    pupil_center = (
        (pupil_left[0] + pupil_right[0]) / 2,
        (pupil_left[1] + pupil_right[1]) / 2
    )
    pupil_height = abs(pupil_center[1] - eye_line_y)

    #plt.title(f"정면 보정된 동공 (높이: {pupil_height:.2f}px)")
    plt.axis('off')
    plt.legend()
    #plt.show()

    # 저장
    save_landmarks_row(
        pts,
        pupil_height,
        adjusted_pupil_left_start,
        adjusted_pupil_left_end,
        adjusted_pupil_right_start,
        adjusted_pupil_right_end
    )



# ====== 결과 저장 ======
def save_landmarks_row(points, pupil_height,
                       left_start, left_end,
                       right_start, right_end):
    global img_path, results

    point_names = [...]  # 기존 그대로 유지

    flat = {}
    for i, (x, y) in enumerate(points):
        name = point_names[i] if i < len(point_names) else f"pt{i}"
        flat[f"{name}_x"] = x
        flat[f"{name}_y"] = y

    flat["pupil_height_px"] = pupil_height

    # 왼쪽 동공 보정
    flat["left_pupil_corrected_start_x"] = left_start[0]
    flat["left_pupil_corrected_start_y"] = left_start[1]
    flat["left_pupil_corrected_end_x"] = left_end[0]
    flat["left_pupil_corrected_end_y"] = left_end[1]

    # 오른쪽 동공 보정
    flat["right_pupil_corrected_start_x"] = right_start[0]
    flat["right_pupil_corrected_start_y"] = right_start[1]
    flat["right_pupil_corrected_end_x"] = right_end[0]
    flat["right_pupil_corrected_end_y"] = right_end[1]

    flat["filename"] = os.path.basename(img_path)
    results.append(flat)





# ====== 메인 루프 ======
# ====== 메인 루프 ======
for subdir, dirs, files in os.walk(root_dir):
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)

    for idx, file in enumerate(image_files):
        print(f"\n📊 진행: {idx+1}/{total_images}")  # ✅ 진행 상태 출력

        img_path = os.path.join(subdir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue
        img_copy = img.copy()
        landmarks = []
        current_guides = []
        img_with_guides = None

        print(f"🖼️ 이미지: {img_path}")

        cv2.namedWindow("Click landmarks", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Click landmarks", 1000, 700)
        cv2.imshow("Click landmarks", img)
        cv2.setMouseCallback("Click landmarks", click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and landmarks:
                print("↩️ 마지막 포인트 삭제")
                landmarks.pop()
                redraw_landmarks()
            elif key == 27:
                print("❌ 작업 취소")
                cv2.destroyAllWindows()
                break
            elif len(landmarks) == 22:
                break


# ====== CSV 저장 ======

if results:
    df_all = pd.DataFrame(results)
    save_path = os.path.join(root_dir, "0527_amond_landmarks2.csv")
    df_all.to_csv(save_path, index=False)
    print(f"\n✅ 전체 CSV 저장 완료: {save_path}")
