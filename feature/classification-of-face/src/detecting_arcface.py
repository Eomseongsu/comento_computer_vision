# classification-of-face/src/detecting_arcface.py
from pathlib import Path
import glob
import cv2
import numpy as np
from numpy.linalg import norm
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ===================== 설정 =====================
ROOT       = Path(__file__).resolve().parent
WEIGHTS    = (ROOT / "../weights/best.pt").resolve()
REF_DIR    = (ROOT / "../data/ref").resolve()     # 특정 인물 참조 이미지 폴더
TEST_DIR   = (ROOT / "../data/test").resolve()    # 추론 대상 이미지 폴더
SAVE_ROOT  = (ROOT / "../runs_pred").resolve()    # 결과 저장 폴더

# YOLO
CONF_YOLO   = 0.25
IOU_YOLO    = 0.60
IMGSZ_YOLO  = 768
DEVICE_YOLO = "cpu"   # GPU면 "0"

# ArcFace/매칭
SIM_THRESH  = 0.33    # (0~1) 미검 많으면 ↓(0.32), 오탐 많으면 ↑(0.35~0.40)
ARC_CTX_ID  = -1      # CPU=-1, GPU 0번=0 (onnxruntime-gpu 필요)
# =================================================

MIN_CROP_W, MIN_CROP_H = 80, 80  # 너무 작은 크롭은 스킵


def build_face_app(ctx_id: int = ARC_CTX_ID) -> FaceAnalysis:
    """
    antelopev2 -> buffalo_l 순으로 시도.
    detection/recognition 모듈만 로드, CPU 실행 기본.
    """
    candidates = ["antelopev2", "buffalo_l"]
    last_err = None
    for name in candidates:
        try:
            print(f"[INFO] Trying InsightFace model pack: {name}")
            app = FaceAnalysis(
                name=name,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"]  # GPU 사용 시 "CUDAExecutionProvider" 추가
            )
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            mkeys = set(getattr(app, "models", {}).keys())
            if not {"detection", "recognition"}.issubset(mkeys):
                raise RuntimeError(f"missing modules: {mkeys}")
            print(f"[INFO] Loaded InsightFace pack: {name} with modules {mkeys}")
            return app
        except Exception as e:
            print(f"[WARN] load {name} failed: {e}")
            last_err = e
    raise RuntimeError(f"All InsightFace packs failed. last error: {last_err}")


def embed_faces(app: FaceAnalysis, bgr):
    """faces와 해당 임베딩 리스트(이미 L2 정규화됨)를 반환"""
    faces = app.get(bgr)
    vecs = [f.normed_embedding for f in faces]
    return vecs, faces


def load_gallery_vecs(app: FaceAnalysis, ref_dir: Path):
    """갤러리의 여러 이미지를 임베딩 벡터 리스트로 로드"""
    vecs = []
    for p in sorted(ref_dir.glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            continue
        im = cv2.imread(str(p))
        if im is None:
            print(f"[WARN] ref read fail: {p}")
            continue
        vlist, _ = embed_faces(app, im)
        if not vlist:
            print(f"[WARN] no face in ref: {p}")
        else:
            vecs.extend(vlist)
    if not vecs:
        raise RuntimeError("No faces found in ref/ gallery. Put clear face images under data/ref/")
    return vecs


def max_cosine_sim(vec: np.ndarray, gallery_vecs: list[np.ndarray]) -> float:
    return float(max(np.dot(vec, g) for g in gallery_vecs))


def list_test_images(test_dir: Path):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"):
        paths += glob.glob(str(test_dir / ext))
    return sorted(paths)


def crop_from_xyxy(img, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return img[y1:y2, x1:x2]


def bbox_contains_point(b, pt):
    x1, y1, x2, y2 = b
    x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def main():
    # 경로/준비
    assert WEIGHTS.exists(), f"weights not found: {WEIGHTS}"
    assert REF_DIR.exists(),  f"ref dir not found: {REF_DIR}"
    assert TEST_DIR.exists(), f"test dir not found: {TEST_DIR}"
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    # 모델
    yolo = YOLO(str(WEIGHTS))
    face_app = build_face_app(ctx_id=ARC_CTX_ID)

    # 갤러리 로드
    gallery_vecs = load_gallery_vecs(face_app, REF_DIR)
    print(f"[INFO] Gallery loaded ({len(gallery_vecs)} vectors) from {REF_DIR}  |  SIM_THRESH={SIM_THRESH}")

    # 테스트 이미지 나열
    img_paths = list_test_images(TEST_DIR)
    print(f"[INFO] test images: {len(img_paths)}")

    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            print(f"[WARN] read fail: {ip}")
            continue

        # 1) YOLO 탐지
        res = yolo.predict(
            source=img, conf=CONF_YOLO, iou=IOU_YOLO, imgsz=IMGSZ_YOLO,
            device=DEVICE_YOLO, verbose=False
        )[0]

        # 2) ArcFace 매칭 (head → person → 전체 이미지 보조)
        kept_indices = []

        # (a) head 우선 (cls=0)
        for i, b in enumerate(res.boxes):
            if int(b.cls.item()) != 0:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
                continue
            crop = crop_from_xyxy(img, (x1, y1, x2, y2))
            if crop is None:
                continue
            vlist, faces = embed_faces(face_app, crop)
            if not vlist:
                continue
            sims = [max_cosine_sim(v, gallery_vecs) for v in vlist]
            if sims and max(sims) >= SIM_THRESH:
                kept_indices.append(i)

        # (b) 보조: person (cls=1)
        for i, b in enumerate(res.boxes):
            if i in kept_indices:
                continue
            if int(b.cls.item()) != 1:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
                continue
            crop = crop_from_xyxy(img, (x1, y1, x2, y2))
            if crop is None:
                continue
            vlist, faces = embed_faces(face_app, crop)
            if not vlist:
                continue
            sims = [max_cosine_sim(v, gallery_vecs) for v in vlist]
            if sims and max(sims) >= SIM_THRESH:
                kept_indices.append(i)

        # (c) 둘 다 실패하면: 전체 이미지에서 얼굴 찾고, 해당 얼굴 중심을 포함하는 박스 선택
        if not kept_indices and len(res.boxes) > 0:
            vlist, faces = embed_faces(face_app, img)
            if faces:
                # 각 얼굴에 대해 유사도 계산
                for f in faces:
                    sim = max_cosine_sim(f.normed_embedding, gallery_vecs)
                    if sim < SIM_THRESH:
                        continue

                    # 얼굴 중심점 계산
                    fx1, fy1, fx2, fy2 = f.bbox.astype(int).tolist()
                    cx = int((fx1 + fx2) / 2); cy = int((fy1 + fy2) / 2)

                    head_candidates, person_candidates = [], []
                    for i, b in enumerate(res.boxes):
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        # 중심점 기준 포함(간단) 또는 IoU로 겹침 판단(정밀)
                        if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                            if int(b.cls.item()) == 0: head_candidates.append(i)
                            elif int(b.cls.item()) == 1: person_candidates.append(i)

                    # head 우선, 없으면 person
                    for cand_list in (head_candidates, person_candidates):
                        for idx in cand_list:
                            if idx not in kept_indices:
                                kept_indices.append(idx)

        # 3) 선택 박스만 유지
        if kept_indices:
            kept = [res.boxes.data[i] for i in kept_indices]
            res.boxes.data = torch.stack(kept, dim=0)
        else:
            res.boxes = res.boxes[:0]

        # 4) 바로 저장만 (표시 없음)
        vis = res.plot()
        out = SAVE_ROOT / f"{Path(ip).stem}_target.jpg"
        cv2.imwrite(str(out), vis)
        print(f"[SAVED] {out}")

    print("[DONE] all results saved in:", SAVE_ROOT)


if __name__ == "__main__":
    main()
