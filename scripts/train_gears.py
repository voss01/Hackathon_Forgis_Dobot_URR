"""
Download the Gears dataset from Roboflow and fine-tune a YOLOv8 model.
Trained weights are copied to backend/weights/gears.pt automatically.

Usage:
    python3 scripts/train_gears.py --api-key <YOUR_ROBOFLOW_API_KEY>

Options:
    --api-key     Roboflow private API key (required)
    --epochs      Number of training epochs       (default: 50)
    --imgsz       Input image size                (default: 640)
    --base-model  YOLOv8 variant to start from    (default: yolov8n.pt)
    --device      Training device: 0 (GPU) or cpu (default: 0)
"""

import argparse
import shutil
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--api-key",    required=True,         help="Roboflow API key")
parser.add_argument("--epochs",     type=int, default=50)
parser.add_argument("--imgsz",      type=int, default=640)
parser.add_argument("--base-model", default="yolov8n.pt",  help="YOLOv8 base weights")
parser.add_argument("--device",     default="0",           help="0=GPU, cpu=CPU")
args = parser.parse_args()

# ── Download dataset from Roboflow ────────────────────────────────────────────
from roboflow import Roboflow

print("[roboflow] connecting ...")
rf = Roboflow(api_key=args.api_key)

project = rf.workspace("ain-shams-university-fn5om").project("gears-dataset")
version  = project.versions()[0]          # latest version
dataset  = version.download("yolov8")    # YOLOv8 format

dataset_yaml = Path(dataset.location) / "data.yaml"
print(f"[roboflow] dataset downloaded to: {dataset.location}")

# ── Train ─────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

print(f"\n[train] starting — base: {args.base_model}, epochs: {args.epochs}, device: {args.device}")
model  = YOLO(args.base_model)
result = model.train(
    data    = str(dataset_yaml),
    epochs  = args.epochs,
    imgsz   = args.imgsz,
    device  = args.device,
    project = "runs/gears",
    name    = "train",
    exist_ok= True,
)

# ── Copy best weights to backend ──────────────────────────────────────────────
best_weights = Path("/app/runs/detect/runs/gears/train/weights/best.pt")
dest         = Path("/app/weights/gears.pt")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(best_weights, dest)

print(f"\n[done] best weights → {dest}")
print(f"       Update YOLO_MODEL in docker-compose.yml or .env:")
print(f"         YOLO_MODEL=/app/weights/gears.pt")
print(f"       Then: docker compose up --build backend")
