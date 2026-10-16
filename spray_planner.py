"""Turns detections into selective-spray commands.

Core idea of the project: only spray where a *weed* is detected, and never where a
crop box overlaps the target. Pure Python/NumPy so it can be unit-tested without a model.
"""
from dataclasses import dataclass
from typing import List, Tuple

WEED, CROP = 1, 0  # keep in sync with CLASS_NAMES in prepare_dataset.py


@dataclass
class Det:
    cls: int
    conf: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def plan_spray(dets: List[Det], img_w=512, n_nozzles=8, min_conf=0.4, max_crop_overlap=0.2):
    """Return (set of nozzle indices to open, list of weed boxes accepted)."""
    crops = [d for d in dets if d.cls == CROP and d.conf >= min_conf]
    targets = []
    for d in dets:
        if d.cls != WEED or d.conf < min_conf:
            continue
        if any(iou(d.box, c.box) > max_crop_overlap for c in crops):
            continue  # too close to a crop -> skip to avoid pesticide on crop
        targets.append(d)
    # One nozzle lane per weed: the lane containing the weed's horizontal centre.
    # Lanes are never opened for a weed that overlaps a crop (filtered above).
    width = img_w / n_nozzles
    nozzles = {min(int(((t.box[0] + t.box[2]) / 2) // width), n_nozzles - 1) for t in targets}
    return sorted(nozzles), targets


if __name__ == "__main__":
    demo = [Det(CROP, .9, (50, 100, 120, 200)), Det(WEED, .8, (300, 120, 360, 180)),
            Det(WEED, .7, (60, 110, 110, 190))]
    print(plan_spray(demo))
