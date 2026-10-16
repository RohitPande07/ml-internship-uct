"""Run inference + spray planning on an image/folder/video.
Usage: python src/predict.py --weights outputs/crop_weed/weights/best.pt --source data/yolo/images/test
"""
import argparse
from ultralytics import YOLO
from spray_planner import Det, plan_spray

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--source", required=True)
ap.add_argument("--conf", type=float, default=0.4)
a = ap.parse_args()

model = YOLO(a.weights)
for r in model.predict(a.source, conf=a.conf, save=True, project="outputs", name="predictions", stream=True):
    dets = [Det(int(c), float(p), tuple(map(float, b))) for c, p, b in
            zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    nozzles, targets = plan_spray(dets, img_w=r.orig_shape[1], min_conf=a.conf)
    print(f"{r.path}: {len(targets)} weeds -> open nozzles {nozzles}")
