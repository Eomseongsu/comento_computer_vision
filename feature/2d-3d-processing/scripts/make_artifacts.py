import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.processing import load_process_and_save

if __name__ == "__main__":
    in_path = os.path.join(ROOT, "data", "sample.jpg")
    out_dir = os.path.join(ROOT, "data", "preprocessed_samples")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"입력 이미지가 없습니다: {in_path}")

    saved = load_process_and_save(in_path, out_dir)
    print(f"saved: {saved}")
