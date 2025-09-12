import os
import cv2
import numpy as np

image = cv2.imread("sample.jpg")
OUTPUT_DIR = "./preprocessed_samples"

# ===== 1.크기조정(224×224) =====
resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sample1.jpg"), resized)

# ===== 2.색상변환(Grayscale & Normalize 적용) =====
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray_norm = gray.astype(np.float32) / 255.0
cv2.imwrite(os.path.join(OUTPUT_DIR, "sample2.jpg"), gray_norm)

# ===== 3.노이즈제거(Blur 필터적용) =====
blur = cv2.GaussianBlur(image, (9, 9), 5)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sample3.jpg"), blur)

# ===== 4.데이터증강(좌우반전, 회전, 색상변화) =====
aug_flip = cv2.flip(resized, 1)
aug_rot = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
aug_bc = cv2.convertScaleAbs(resized, alpha=1.0, beta=-50)
aug_combined = np.hstack([aug_flip, aug_rot, aug_bc])
cv2.imwrite(os.path.join(OUTPUT_DIR, "sample4.jpg"), aug_combined)

# ===== 5. 너무 어두운 이미지 제거 (평균 밝기 기준) =====
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
mean_brightness = float(gray.mean())
dark_flag = mean_brightness >= 40
print(f"밝기 = {mean_brightness:.1f}")

if dark_flag:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sample5.jpg"), resized)

print("실행 완료.")