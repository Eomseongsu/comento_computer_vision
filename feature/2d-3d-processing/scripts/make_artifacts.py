import os
import sys
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.processing import (
    load_process_and_save,
    generate_depth_map,
    save_point_cloud_as_ply,
)

if __name__ == "__main__":
    in_path = os.path.join(ROOT, "data", "sample.jpg")
    out_dir = os.path.join(ROOT, "data", "preprocessed_samples")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"입력 이미지가 없습니다: {in_path}")

    # 1) Depth Map 산출 & 저장 (JPG)
    saved_depth = load_process_and_save(in_path, out_dir)
    print(f"saved depth map: {saved_depth}")

    # 2) 포인트클라우드 생성 & 저장 (PLY)
    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    depth_map, points_3d = generate_depth_map(image)
    ply_path = save_point_cloud_as_ply(points_3d, colors_bgr=depth_map, out_dir=out_dir)
    print(f"saved point cloud (PLY): {ply_path}")
