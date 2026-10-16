"""Split the 1300-image YOLO dataset into train/val/test and write data.yaml.

Expected raw layout (images and YOLO .txt labels side by side or in two folders):
    data/raw/images/*.jpg   data/raw/labels/*.txt
Usage: python src/prepare_dataset.py --raw data/raw --out data/yolo
Class ids (edit CLASS_NAMES if your labels differ): 0 = crop (sesame), 1 = weed
"""
import argparse, random, shutil
from pathlib import Path
import yaml

CLASS_NAMES = ["crop", "weed"]
EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    img_dir = raw / "images" if (raw / "images").is_dir() else raw
    lbl_dir = raw / "labels" if (raw / "labels").is_dir() else raw

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXT)
    pairs = [(i, lbl_dir / f"{i.stem}.txt") for i in imgs if (lbl_dir / f"{i.stem}.txt").exists()]
    print(f"{len(imgs)} images, {len(pairs)} with labels")
    if not pairs:
        raise SystemExit("No image/label pairs found.")

    # Augmented copies of one photo must not straddle splits; group by stem prefix
    # before '_aug' if present, so leakage is avoided.
    groups = {}
    for i, l in pairs:
        groups.setdefault(i.stem.split("_aug")[0], []).append((i, l))
    keys = sorted(groups); random.Random(a.seed).shuffle(keys)
    n = len(keys); c1 = int(a.split[0] * n); c2 = c1 + int(a.split[1] * n)
    parts = {"train": keys[:c1], "val": keys[c1:c2], "test": keys[c2:]}

    for split, ks in parts.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for k in ks:
            for i, l in groups[k]:
                shutil.copy(i, out / "images" / split / i.name)
                shutil.copy(l, out / "labels" / split / l.name)
        print(split, sum(len(groups[k]) for k in ks), "images")

    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
