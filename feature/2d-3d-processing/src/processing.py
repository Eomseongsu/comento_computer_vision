import os
import cv2
import numpy as np
from typing import Tuple, Optional


def generate_depth_map(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지를 그레이스케일로 변환 후 JET 컬러맵을 적용한 Depth Map(BGR, uint8)과
    의사 3D 포인트 클라우드(X,Y,Z)를 반환합니다.

    반환:
      - depth_map: (H,W,3) BGR uint8 (시각화용 컬러맵)
      - points_3d: (H,W,3) float32, 각 픽셀의 (X,Y,Z)
                   여기서 Z는 실제 깊이가 아닌 '밝기값(0~255)'
    """
    if image is None:
        raise ValueError("입력 이미지가 None 입니다.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("입력 이미지는 (H,W,3) BGR 이어야 합니다.")

    # 1) 그레이스케일
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) 시각화용 컬러맵(Depth Map)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # (H,W,3) uint8

    # 3) 포인트클라우드 (X,Y,Z) 계산 (Z는 밝기값)
    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    Z = gray.astype(np.float32)
    points_3d = np.dstack((X, Y, Z)).astype(np.float32)  # (H,W,3)

    return depth_map, points_3d


def save_depth_map(depth_map: np.ndarray, out_dir: str, filename: str = "3d_sample.jpg") -> str:
    """Depth Map 이미지를 디스크에 저장하고 저장 경로를 반환합니다."""
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    ok = cv2.imwrite(save_path, depth_map)
    if not ok:
        raise IOError(f"이미지 저장 실패: {save_path}")
    return save_path


def save_point_cloud_as_ply(points_3d: np.ndarray,
                            colors_bgr: Optional[np.ndarray],
                            out_dir: str,
                            filename: str = "points_pseudo_depth.ply") -> str:
    """
    (H,W,3) 포인트클라우드와 색상을 받아 PLY(ASCII)로 저장합니다.
    colors_bgr가 None이 아니면 RGB로 변환하여 색도 같이 저장합니다.
    """
    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError("points_3d는 (H,W,3) 이어야 합니다.")
    h, w, _ = points_3d.shape
    pts = points_3d.reshape(-1, 3)  # (N,3)

    cols_rgb = None
    if colors_bgr is not None:
        if colors_bgr.shape[:2] != (h, w):
            raise ValueError("colors_bgr의 H,W가 points_3d와 일치해야 합니다.")
        cols_bgr = colors_bgr.reshape(-1, 3)
        cols_rgb = cols_bgr[:, ::-1]  # BGR -> RGB

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)

    with open(save_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols_rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        if cols_rgb is not None:
            for (x, y, z), (r, g, b) in zip(pts, cols_rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z) in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    return save_path


def load_process_and_save(input_path: str,
                          out_dir: str = "../data/preprocessed_samples") -> str:
    """
    파일 경로에서 이미지를 읽어 depth map을 만들고 저장까지 수행.
    Depth Map 저장 후 저장 경로를 반환합니다.
    """
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지 로드 실패: {input_path}")

    depth_map, _ = generate_depth_map(image)
    return save_depth_map(depth_map, out_dir)
