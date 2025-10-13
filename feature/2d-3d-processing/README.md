# 2D → 3D Processing (Depth Map + Unit Test)

## 1) 개요
- **기능**: 2D 이미지를 그레이스케일로 변환한 뒤 JET 컬러맵을 적용해 **Depth Map**을 생성합니다. 필요 시 결과 이미지를 디스크에 저장합니다.

---

## 2) 폴더 구조
```
2d-3d-processing/
├─ .pytest_cache/                         # pytest 캐시 (자동 생성)
├─ data/
│   ├─ preprocessed_samples/3d_sample.jpg # 산출물 (스크립트 실행 시 생성)
│   └─ sample.jpg                         # 입력 이미지(사용자 준비)
├─ scripts/
│   └─ make_artifacts.py                  # 산출물 생성 스크립트
├─ src/
│   ├─ __init__.py                        # 패키지 인식용
│   └─ processing.py                      # 핵심 로직 (생성/저장 함수)
├─ tests/
│   └─ test_3d_processing.py              # 유닛 테스트
└─ pytest.ini                             # pytest 설정(pythonpath, testpaths)
```
> `__pycache__`, `.pytest_cache`는 자동 생성되는 캐시입니다. 저장소에 올리지 않아도 됩니다.

---

## 3) 환경
- Python **3.10+**
- 패키지 설치:
```bash
pip install numpy opencv-python pytest
```

---

## 4) 설정 (pytest.ini)
루트의 `pytest.ini`에서 `pythonpath = src`, `testpaths = tests`를 지정해 테스트 시 `src`를 자동 검색 경로에 추가합니다.

```ini
[pytest]
pythonpath = src
testpaths  = tests
```

---

## 5) 핵심 기능 (src/processing.py)
- `generate_depth_map(image) -> np.ndarray`  
  - BGR 이미지를 그레이스케일로 변환 후 **COLORMAP_JET** 적용 (BGR, `uint8`, `(H,W,3)`).
- `save_depth_map(depth_map, out_dir, filename='3d_sample.jpg') -> str`  
  - 출력 디렉토리를 생성한 뒤 `cv2.imwrite()`로 저장. 경로를 반환.
- `load_process_and_save(input_path, out_dir) -> str`  
  - 파일에서 이미지를 읽어 depth map 생성 후 저장까지 한 번에 수행.

---

## 6) 테스트 실행 (Unit Test)
**루트에서 실행**하세요. (`pytest.ini`가 `pythonpath=src`, `testpaths=tests`를 지정)
```bash
pytest -q
# 또는 특정 파일만
pytest -q tests/test_3d_processing.py
```

### 예시 출력
```
..                                                              [100%]
2 passed in 0.20s
```
> 테스트는 **임시 디렉터리(tmp_path)** 를 사용하므로 **프로젝트 폴더에 산출물을 남기지 않습니다.**

---

## 7) 산출물 생성 (Depth Map 이미지 저장)
테스트와 별도로, 실제 파일을 생성하려면 **스크립트**를 실행하세요.

```bash
# 루트에서 입력 이미지가 존재하는지 확인: ./data/sample.jpg
python scripts\make_artifacts.py
```

- 파일: `data/preprocessed_samples/3d_sample.jpg` 생성

> `scripts/make_artifacts.py`는 실행 시 리포지토리 루트를 `sys.path`에 추가하여 `from src.processing import ...` 임포트를 보장합니다. 실행은 **항상 루트에서** 하세요.

---

## 8) 트러블슈팅 (FAQ)
**Q1. `ModuleNotFoundError: No module named 'src'`**  
- 루트에서 실행하고 있는지 확인 (`pytest -q`, `python scripts/make_artifacts.py`)  
- `pytest.ini`에 `pythonpath = src`가 있는지 확인  
- (스크립트 실행 시) `scripts/make_artifacts.py`가 `sys.path`에 루트를 추가하는 코드가 포함되어 있는지 확인

**Q2. 테스트는 통과했는데 리포지토리에 이미지는 안 생겨요.**  
- 정상입니다. 테스트는 임시 디렉터리를 사용합니다. 실제 산출물은 `scripts/make_artifacts.py`로 생성하세요.

---

## 9) 캐시 폴더 정리
```bash
# 선택: 캐시 제거
# PowerShell
Remove-Item -Recurse -Force __pycache__, .pytest_cache
# CMD
rmdir /s /q __pycache__ .pytest_cache
# Git Bash/WSL
rm -rf __pycache__ .pytest_cache
```

---
