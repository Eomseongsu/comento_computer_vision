🖼️ 1차 업무

AI 학습을 위한 이미지 전처리 파이프라인 예시 코드입니다.
주어진 이미지를 불러와 크기 조정 → 색상 변환 → 노이즈 제거 → 데이터 증강 → 이상치 제거 과정을 거친 결과를 저장합니다.

📂 환경 세팅
요구 패키지

Python 3.8+
OpenCV
NumPy

pip install opencv-python numpy

⚙️ 실행 방법

sample.jpg 이미지를 프로젝트 루트에 준비

스크립트 실행:

python image_preprocessing.py


결과 이미지는 ./preprocessed_samples/ 폴더에 생성됩니다.

📝 전처리 단계
단계	설명	결과 파일
1. 크기 조정	입력 이미지를 224×224로 리사이즈	sample1.jpg
2. 색상 변환	BGR → Grayscale 변환, [0,255] → [0,1] 정규화	sample2.jpg (저장 시 거의 검게 보일 수 있음)
3. 노이즈 제거	Gaussian Blur 적용 (커널 9×9, σ=5)	sample3.jpg
4. 데이터 증강	좌우 반전 + 회전(90°) + 밝기 감소 세 장을 가로로 결합	sample4.jpg
5. 이상치 제거	Grayscale 평균 밝기 < 40 → DROP (저장 안 함), ≥ 40 → KEEP	sample5.jpg (조건 만족 시만 저장)
📊 출력 구조 예시
preprocessed_samples/
├── sample1.jpg   # 크기 조정
├── sample2.jpg   # Grayscale & Normalize
├── sample3.jpg   # Blur
├── sample4.jpg   # Augmentation (Flip+Rotate+Brightness)
└── sample5.jpg   # 밝기 기준 필터링 (조건 충족 시)

⚠️ 참고 사항

sample2.jpg (Grayscale Normalize)는 [0,1] 범위 float를 저장하기 때문에 검은색에 가까운 이미지로 보일 수 있음 → 학습용으로는 OK, 시각화할 땐 *255 후 uint8 변환 권장

밝기 임계값(threshold=40)은 데이터셋 분포에 맞게 조정 가능

Augmentation 결과(sample4.jpg)는 Flip/Rotate/Brightness 세 이미지를 한 장으로 비교용으로 저장