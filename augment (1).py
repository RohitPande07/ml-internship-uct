"""Reproduces the data-augmentation step of the data preparation (546 -> ~1300 images).
Flip augmentations (so YOLO boxes can be
transformed exactly) plus brightness/contrast jitter (equivalent in spirit to Keras ImageDataGenerator).

Usage: python src/augment.py --images data/clean/images --labels data/clean/labels --out data/raw --target 1300
"""
import argparse, random
from pathlib import Path
import cv2, numpy as np


def read_labels(p):
    rows = [l.split() for l in p.read_text().strip().splitlines() if l.strip()]
    return [(int(r[0]), *map(float, r[1:5])) for r in rows]


def write_labels(p, boxes):
    p.write_text("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes))


def hflip(img, b):  return img[:, ::-1], [(c, 1 - x, y, w, h) for c, x, y, w, h in b]
def vflip(img, b):  return img[::-1], [(c, x, 1 - y, w, h) for c, x, y, w, h in b]
def both(img, b):   return hflip(*vflip(img, b))


def jitter(img, rng):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)       # contrast / brightness
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--target", type=int, default=1300)
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng, r = np.random.default_rng(a.seed), random.Random(a.seed)
    out_i, out_l = Path(a.out, "images"), Path(a.out, "labels")
    out_i.mkdir(parents=True, exist_ok=True); out_l.mkdir(parents=True, exist_ok=True)

    srcs = [p for p in sorted(Path(a.images).iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and Path(a.labels, p.stem + ".txt").exists()]
    count = 0
    for p in srcs:  # 1) resize originals 4000x3000 -> 512x512 (YOLO labels are normalised, so unchanged)
        img = cv2.resize(cv2.imread(str(p)), (a.size, a.size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_i / f"{p.stem}.jpg"), img)
        write_labels(out_l / f"{p.stem}.txt", read_labels(Path(a.labels, p.stem + ".txt")))
        count += 1
    ops = [hflip, vflip, both]
    while count < a.target:  # 2) fill up to target with augmented copies
        p = r.choice(srcs)
        img = cv2.imread(str(out_i / f"{p.stem}.jpg")); b = read_labels(out_l / f"{p.stem}.txt")
        img, b = r.choice(ops)(img, b)
        img = jitter(np.ascontiguousarray(img), rng)
        name = f"{p.stem}_aug{count}"
        cv2.imwrite(str(out_i / f"{name}.jpg"), img); write_labels(out_l / f"{name}.txt", b)
        count += 1
    print(f"Wrote {count} images to {a.out}")


if __name__ == "__main__":
    main()
