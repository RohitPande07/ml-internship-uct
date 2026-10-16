"""Train YOLOv8 on the crop/weed dataset.
Usage: python src/train.py --data data/yolo/data.yaml --model yolov8n.pt --epochs 100
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None)
a = ap.parse_args()

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project="outputs", name="crop_weed", patience=20, seed=42)
m = model.val(data=a.data, split="test")
print(f"Test mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")
