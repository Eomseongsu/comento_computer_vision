# Classification of Face (YOLOv8 + ArcFace)

3차 업무에서 만든 **`object-detecting/`** 폴더의 **모델(`best.pt`)과 이미지**를 가져와, 특정 인물만 식별하도록 확장한 프로젝트입니다.  
YOLOv8(내가 학습한 `best.pt`)으로 사람/머리를 감지하고, **ArcFace(InsightFace)** 임베딩으로 **지정한 사람만 필터링**합니다.

---

## 폴더 구조

```
classification-of-face/
├─ data/
│  ├─ ref/            # 특정 인물 참조 이미지 10장 (다양 각도/조명 권장)
│  └─ test/           # 테스트 이미지 3장
├─ runs_pred/         # 결과 이미지 3장 (실행 후 저장됨)
├─ src/
│  └─ detecting_arcface.py
└─ weights/
   └─ best.pt         # object-detecting에서 가져온 학습 가중치
```

---

## 환경 세팅

> 권장: 가상환경(venv) 사용

```bash
cd classification-of-face

# 1) venv 생성 및 활성화 (Windows PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1

# 2) 필수 패키지 설치 (NumPy 1.x 고정으로 충돌 피함)
pip install -U pip wheel setuptools
pip install "numpy==1.26.4"
pip install "opencv-python==4.9.0.80" "onnxruntime==1.18.1" "scikit-image==0.21.0" "matplotlib==3.7.5"
pip install "ultralytics==8.3.0" "insightface==0.7.3"

# (GPU로 ArcFace 실행하려면)
# pip install onnxruntime-gpu==1.18.1
```
> venv 끄기: `deactivate`  
> 다시 켜기: `. .venv\Scripts\Activate.ps1`

---

## 사용 방법

1) **참조 이미지 준비**:  
   `data/ref/`에 **특정 인물 사진 10장**(정면/측면/밝기·각도 다양)을 넣습니다.
2) **테스트 이미지**:  
   `data/test/`에 테스트 이미지 3장을 둡니다.
3) **실행**:
   ```bash
   python src/detecting_arcface.py
   ```
4) **결과 확인**:  
   `runs_pred/`에 `_target.jpg`가 저장됩니다(3장).

---

## 동작 개요

- **검출**: YOLOv8(`weights/best.pt`)이 `head(0) / person(1)` 박스를 생성  
- **식별(ArcFace)**: 참조(ref) 이미지 임베딩들과 **코사인 유사도** 비교 → 임계치 이상만 유지  
- **다중 인물 대응**: 한 이미지에 동일 인물이 여러 명 있어도 **모두 인식**되도록 처리  
- **저장 모드**: 화면 표시 없이 바로 `runs_pred/`에 저장

---

## 주요 파라미터 (src/detecting_arcface.py)

- `SIM_THRESH` (기본 `0.33`): 유사도 임계치.  
  - **미검 많음** → `0.32`로 ↓  
  - **오탐 많음** → `0.35~0.40`로 ↑
- `IMGSZ_YOLO` (기본 `768`): 작은 얼굴이 많다면 ↑ (속도↓, 검출↑)
- `ARC_CTX_ID` (기본 `-1`): ArcFace 실행 장치. CPU=`-1`, GPU=`0`(onnxruntime-gpu 필요)

---

## 참고

- `weights/best.pt`는 **object-detecting** 프로젝트에서 학습된 가중치입니다.  
- 데이터/가중치 라이선스 및 사용 정책은 원본 프로젝트 정책을 따릅니다.
