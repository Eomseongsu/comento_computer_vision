# classification-of-face/src/detecting_arcface.py
from pathlib import Path
import glob, argparse, json, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# -------------------- 기본 경로 --------------------
ROOT      = Path(__file__).resolve().parent
WEIGHTS   = (ROOT / "../weights/best.pt").resolve()
REF_DIR   = (ROOT / "../data/ref").resolve()
TEST_DIR  = (ROOT / "../data/test").resolve()
SAVE_ROOT = (ROOT / "../runs_pred").resolve()
CACHE_DIR = (ROOT / "../.cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- 유틸 --------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1); area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def list_images(p: Path):
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff"):
        paths += glob.glob(str(p / ext))
    return sorted(paths)

# -------------------- ArcFace 로더(탄력) --------------------
def build_face_app(ctx_id=-1, providers=("CPUExecutionProvider",)):
    for name in ("antelopev2", "buffalo_l"):
        try:
            print(f"[INFO] Loading InsightFace pack: {name}")
            app = FaceAnalysis(
                name=name,
                allowed_modules=["detection","recognition"],
                providers=list(providers)
            )
            app.prepare(ctx_id=ctx_id, det_size=(640,640))
            m = set(getattr(app, "models", {}).keys())
            if not {"detection","recognition"}.issubset(m):
                raise RuntimeError(f"missing modules {m}")
            return app
        except Exception as e:
            print(f"[WARN] fallback: {name} failed: {e}")
    raise RuntimeError("All InsightFace packs failed.")

def embed_faces(app: FaceAnalysis, bgr):
    faces = app.get(bgr)
    return [f.normed_embedding for f in faces], faces

# -------------------- 갤러리 캐시 --------------------
def cache_path_for_gallery(ref_dir: Path, who="default"):
    sig = str(ref_dir.resolve())
    return CACHE_DIR / f"gallery_{who}_{abs(hash(sig))%10**10}.npy"

def load_gallery(app: FaceAnalysis, ref_dir: Path, who="default"):
    cp = cache_path_for_gallery(ref_dir, who)
    if cp.exists():
        try:
            arr = np.load(cp)
            if arr.ndim == 2:  # (N,512)
                print(f"[INFO] Gallery cache loaded: {cp} ({arr.shape[0]} vecs)")
                return arr
        except Exception:
            pass

    vecs = []
    for p in list_images(ref_dir):
        im = cv2.imread(p)
        if im is None: 
            print(f"[WARN] fail to read ref: {p}"); 
            continue
        vlist, _ = embed_faces(app, im)
        if not vlist: 
            print(f"[WARN] no face in ref: {p}")
        else:
            vecs.extend(vlist)
    if not vecs:
        raise RuntimeError("No faces in ref/. Put clear faces there.")

    arr = np.stack(vecs, 0).astype(np.float32)
    np.save(cp, arr)
    print(f"[INFO] Gallery built & cached: {cp} ({arr.shape[0]} vecs)")
    return arr

def max_cosine_sim(vec, gallery_arr: np.ndarray):
    # gallery_arr: (N,512), vec: (512,)
    return float(np.max(gallery_arr @ vec))

# -------------------- 파이프라인 --------------------
def run(
    imgsz=768, conf_th=0.25, iou_nms=0.60,
    sim_th=0.33, iou_face_overlap=0.10,
    min_crop_wh=80, device_yolo="cpu",
    ctx_id_arc=-1, providers=("CPUExecutionProvider",),
    multi_targets: dict|None=None   # {"alice": Path, "bob": Path} 형태 지원
):
    assert WEIGHTS.exists(), f"weights not found: {WEIGHTS}"
    assert TEST_DIR.exists(), f"test dir not found: {TEST_DIR}"

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(WEIGHTS))
    face_app = build_face_app(ctx_id=ctx_id_arc, providers=providers)

    # --- 갤러리 구성 ---
    target_galleries = {}
    if multi_targets:  # 여러 사람 화이트리스트
        for name, ref in multi_targets.items():
            target_galleries[name] = load_gallery(face_app, Path(ref), who=name)
    else:              # 단일 타깃 (기본: data/ref)
        target_galleries["target"] = load_gallery(face_app, REF_DIR, who="target")

    img_paths = list_images(TEST_DIR)
    print(f"[INFO] test images: {len(img_paths)}")

    summary = []
    t0 = time.time()
    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            print(f"[WARN] read fail: {ip}")
            continue

        res = yolo.predict(
            source=img, conf=conf_th, iou=iou_nms, imgsz=imgsz,
            device=device_yolo, verbose=False
        )[0]

        kept_idxs = []
        box_sims  = {}  # idx -> (name, best_sim)

        # (a) head 우선
        for i, b in enumerate(res.boxes):
            if int(b.cls.item()) != 0:  # 0=head
                continue
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            if (x2-x1) < min_crop_wh or (y2-y1) < min_crop_wh:
                continue
            crop = img[y1:y2, x1:x2]
            vlist, _ = embed_faces(face_app, crop)
            if not vlist: 
                continue
            best_name, best_sim = None, -1.0
            for v in vlist:
                for name, gal in target_galleries.items():
                    s = max_cosine_sim(v, gal)
                    if s > best_sim:
                        best_sim, best_name = s, name
            if best_sim >= sim_th:
                kept_idxs.append(i); box_sims[i]=(best_name, best_sim)

        # (b) person 보조
        for i, b in enumerate(res.boxes):
            if i in kept_idxs or int(b.cls.item()) != 1:  # 1=person
                continue
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            if (x2-x1) < min_crop_wh or (y2-y1) < min_crop_wh:
                continue
            crop = img[y1:y2, x1:x2]
            vlist, _ = embed_faces(face_app, crop)
            if not vlist: 
                continue
            best_name, best_sim = None, -1.0
            for v in vlist:
                for name, gal in target_galleries.items():
                    s = max_cosine_sim(v, gal)
                    if s > best_sim:
                        best_sim, best_name = s, name
            if best_sim >= sim_th:
                kept_idxs.append(i); box_sims[i]=(best_name, best_sim)

        # (c) 전체 이미지 보조: 모든 얼굴 검사 → IoU로 겹치는 박스 추가
        if len(res.boxes) > 0:
            vlist, faces = embed_faces(face_app, img)
            for f in faces:
                # 타깃 중 최댓값
                best_name, best_sim = None, -1.0
                for name, gal in target_galleries.items():
                    s = max_cosine_sim(f.normed_embedding, gal)
                    if s > best_sim:
                        best_sim, best_name = s, name
                if best_sim < sim_th: 
                    continue
                fx1, fy1, fx2, fy2 = f.bbox.astype(int).tolist()
                face_box = (fx1, fy1, fx2, fy2)
                head_cands, person_cands = [], []
                for i, b in enumerate(res.boxes):
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    ov = iou_xyxy(face_box, (x1,y1,x2,y2))
                    if ov >= iou_face_overlap:
                        if int(b.cls.item()) == 0: head_cands.append(i)
                        elif int(b.cls.item()) == 1: person_cands.append(i)
                for lst in (head_cands, person_cands):
                    for idx in lst:
                        if idx not in kept_idxs:
                            kept_idxs.append(idx); box_sims[idx]=(best_name, best_sim)

        # 선택 박스만 유지 + 점수/이름 오버레이
        if kept_idxs:
            kept = [res.boxes.data[i] for i in kept_idxs]
            res.boxes.data = torch.stack(kept, 0)
        else:
            res.boxes = res.boxes[:0]

        vis = res.plot()
        # label 덧그리기(유사도)
        for i, b in enumerate(res.boxes):
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            nm, sm = box_sims.get(kept_idxs[i], ("target", 0.0))
            txt = f"{nm}:{sm:.2f}"
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(vis, txt, (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out = SAVE_ROOT / f"{Path(ip).stem}_target.jpg"
        cv2.imwrite(str(out), vis)
        summary.append({"image": ip, "kept": len(kept_idxs)} )
        print(f"[SAVED] {out} (kept={len(kept_idxs)})")

    with open(SAVE_ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved -> {SAVE_ROOT} | time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_nms", type=float, default=0.60)
    ap.add_argument("--sim", type=float, default=0.33)
    ap.add_argument("--iou_face", type=float, default=0.10)
    ap.add_argument("--min_crop", type=int, default=80)
    ap.add_argument("--device", type=str, default="cpu")     # "0" for GPU
    ap.add_argument("--arc_ctx", type=int, default=-1)       # 0 for GPU(onnxruntime-gpu)
    ap.add_argument("--multi", type=str, default=None, 
                    help='JSON: {"alice":"../data/ref_alice","bob":"../data/ref_bob"}')
    args = ap.parse_args()

    multi = json.loads(args.multi) if args.multi else None
    run(
        imgsz=args.imgsz, conf_th=args.conf, iou_nms=args.iou_nms,
        sim_th=args.sim, iou_face_overlap=args.iou_face, min_crop_wh=args.min_crop,
        device_yolo=args.device, ctx_id_arc=args.arc_ctx,
        providers=("CPUExecutionProvider",), multi_targets=multi
    )
