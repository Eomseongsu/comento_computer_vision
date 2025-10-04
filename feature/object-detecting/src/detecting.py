# src/detecting.py
from ultralytics import YOLO
from pathlib import Path
import cv2, numpy as np

# 경로 설정 (프로젝트 루트 기준)
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "../weights/best.pt"
IMAGES  = [ROOT/"../data/image1.jpg", ROOT/"../data/image2.jpg", ROOT/"../data/image3.jpg"]

def preprocess(img, gamma=1.0, maxw=1600):
    # 가벼운 전처리: 리사이즈 + 감마(밝기) 보정
    h, w = img.shape[:2]
    if w > maxw:
        img = cv2.resize(img, (maxw, int(h*maxw/w)), interpolation=cv2.INTER_AREA)
    if gamma != 1.0:
        table = ((np.arange(256)/255.0)**(1.0/gamma)*255).astype(np.uint8)
        img = cv2.LUT(img, table)
    return img

def main():
    model = YOLO(str(WEIGHTS))
    print("[INFO] S: 저장,  아무 키: 다음,  Q/ESC: 종료")

    for p in IMAGES:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] 읽기 실패: {p}"); continue

        pre = preprocess(img, gamma=1.0)  # 필요 시 0.9~1.2로 조정
        res = model.predict(source=pre, conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
        vis = res.plot()  # 박스/라벨이 그려진 BGR 이미지

        cv2.imshow("YOLOv8", vis)
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')): break
        if k in (ord('s'),):    # 저장
            out = ROOT / "../runs_pred" / f"{Path(p).stem}_pred.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
            print("[SAVED]", out)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
