# Unit Test 실습: 가짜 깊이맵 생성

## 1. 개요
- OpenCV를 활용하여 입력 이미지를 그레이스케일로 변환한 뒤, 컬러맵(JET)을 적용해 **가짜 깊이맵(Depth Map)**을 생성합니다.
- `pytest`를 이용해 해당 함수의 출력 크기와 데이터 타입을 검증하는 **Unit Test**를 구성했습니다.

---

## 2. 환경
- Python 3.10 이상
- 필수 패키지:
  ```bash
  pip install numpy opencv-python pytest
  ```

---

## 3. 코드 설명
### (1) 대상 함수
```python
def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return depth_map
```
- `image=None`일 경우 `ValueError` 발생
- 입력 이미지를 그레이스케일로 변환 후, `COLORMAP_JET` 적용 → 가짜 깊이맵 생성

### (2) Unit Test
```python
def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # 검정색 빈 이미지
    depth_map = generate_depth_map(image)

    # 출력 검증
    assert depth_map.shape== image.shape,
    "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray),
    "출력 데이터 타입이 ndarray가 아닙니다."
```
- 출력 이미지의 크기와 타입을 확인하는 기본적인 테스트

---

## 4. 실행 방법
```bash
pytest unit_test_processing.py
```

---

## 5. 실행 결과
실제 실행 로그:

```
======================= test session starts ========================
platform win32 -- Python 3.10.10, pytest-8.4.2, pluggy-1.6.0
rootdir: E:\comento\comento_computer_vision\feature\unit-test-preprocessing\src
collected 1 item

unit_test_processing.py .                                     [100%]

======================== 1 passed in 0.18s =========================
```

- 점(`.`) = 테스트 1개 성공
- `1 passed` = 모든 테스트가 성공적으로 통과

---

## 6. 실행 후 생성되는 파일/폴더
테스트 실행 후 자동으로 다음과 같은 캐시 파일/폴더가 생깁니다:
```
__pycache__/
└─ unit_test_processing.cpython-310-pytest-8.4.2.pyc

.pytest_cache/
├─ v/cache/nodeids
├─ .gitignore
├─ .CACHEDIR.TAG
└─ README.md
```
- `__pycache__` : Python 바이트코드 캐시  
- `.pytest_cache` : pytest 실행 기록 (마지막 실행 결과, 편의 기능 제공)  
- 필요 없다면 삭제해도 되며, 보통 `.gitignore`에 추가하여 저장소에는 올리지 않습니다.

---

## 7. 결론
- 작성한 Unit Test를 통해 `generate_depth_map()` 함수의 **기본 동작(출력 크기, 타입)**을 자동으로 검증할 수 있었습니다.  
- Unit Test는 코드의 안정성을 보장하고, 함수 수정 시 발생할 수 있는 오류를 **사전에 예방**할 수 있는 중요한 도구임을 확인했습니다.
