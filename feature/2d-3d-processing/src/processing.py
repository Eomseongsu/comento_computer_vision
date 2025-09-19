import os
import cv2
import numpy as np

def generate_depth_map(image: np.ndarray) -> np.ndarray:
    """이미지를 그레이스케일로 변환 후 JET 컬러맵을 적용한 Depth Map(BGR, uint8)을 반환."""
    if image is None:
        raise ValueError("입력 이미지가 None 입니다.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("입력 이미지는 (H,W,3) BGR 이어야 합니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # (H,W,3) uint8
    return depth_map

def save_depth_map(depth_map: np.ndarray, out_dir: str, filename: str = "3d_sample.jpg") -> str:
    """Depth Map 이미지를 디스크에 저장하고 저장 경로를 반환."""
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    ok = cv2.imwrite(save_path, depth_map)
    if not ok:
        raise IOError(f"이미지 저장 실패: {save_path}")
    return save_path

def load_process_and_save(input_path: str, out_dir: str = "../data/preprocessed_samples") -> str:
    """파일 경로에서 이미지를 읽어 depth map을 만들고 저장까지 수행."""
    image = cv2.imread(input_path)
    depth_map = generate_depth_map(image)
    return save_depth_map(depth_map, out_dir)
