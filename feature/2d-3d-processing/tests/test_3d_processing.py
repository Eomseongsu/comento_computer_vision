import numpy as np
import cv2
import os
import pytest
from processing import generate_depth_map, save_depth_map

def test_generate_depth_map():
    """가짜 이미지로 depth map 생성이 정상 동작하는지 확인"""
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # 검정색 (멀리)
    depth_map = generate_depth_map(image)

    assert isinstance(depth_map, np.ndarray), "출력 타입이 ndarray가 아님"
    assert depth_map.shape == image.shape, "출력 크기가 입력과 다름"
    assert depth_map.dtype == np.uint8, "출력 dtype은 uint8이어야 함"
    assert depth_map.ndim == 3 and depth_map.shape[2] == 3, "출력은 BGR 3채널이어야 함"

def test_save_depth_map(tmp_path):
    """생성한 depth map을 디스크에 저장하고 정상 저장 여부를 확인"""
    image = np.full((50, 60, 3), 180, dtype=np.uint8)  # 회색(중간 깊이)
    depth_map = generate_depth_map(image)

    out_dir = tmp_path / "preprocessed_samples"
    save_path = save_depth_map(depth_map, str(out_dir), "3d_sample.jpg")

    assert os.path.exists(save_path), "파일이 저장되지 않음"
    reloaded = cv2.imread(save_path)
    assert reloaded is not None, "저장된 파일을 읽을 수 없음"
    assert reloaded.shape == depth_map.shape, "저장/재로딩 후 크기가 다름"
