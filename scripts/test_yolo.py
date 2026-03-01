"""
YOLO detection test — fetches a live snapshot from the camera API,
runs detection, draws annotated bounding boxes, and saves the result.

Run from the project root:
    docker exec forgis-backend bash -c \
        "source /app/.venv/bin/activate && python3 /scripts/test_yolo.py"

Optional env vars:
    CLASS_FILTER    comma-separated class names to keep  (default: all)
    CONF_THRESHOLD  minimum confidence 0-1               (default: 0.3)
    OUTPUT_PATH     where to save the annotated image    (default: /tmp/yolo_result.jpg)
    SNAPSHOT_URL    camera snapshot endpoint             (default: http://localhost:8000/api/camera/snapshot)
    YOLO_MODEL      path to model weights                (default: /app/weights/roboflow_logistics.pt)
"""

import os
import sys
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("YOLO_MODEL",      "/app/weights/gears.pt")
SNAPSHOT_URL    = os.environ.get("SNAPSHOT_URL",    "http://localhost:8000/api/camera/snapshot")
OUTPUT_PATH     = os.environ.get("OUTPUT_PATH",     "/tmp/yolo_result.jpg")
CONF_THRESHOLD  = float(os.environ.get("CONF_THRESHOLD", "0.7"))
CLASS_FILTER_RAW = os.environ.get("CLASS_FILTER",   "")
CLASS_FILTER    = [c.strip().lower() for c in CLASS_FILTER_RAW.split(",") if c.strip()]

# ── Colour palette (one per class index) ─────────────────────────────────────
PALETTE = [
    (56,  255, 56),   (255, 56,  56),   (56,  56,  255),  (255, 200, 0),
    (0,   200, 255),  (255, 0,   200),  (128, 255, 0),    (255, 128, 0),
    (0,   128, 255),  (128, 0,   255),  (200, 255, 200),  (255, 200, 200),
    (200, 200, 255),  (0,   255, 200),  (200, 0,   255),  (255, 255, 56),
    (56,  255, 255),  (255, 56,  255),  (100, 180, 100),  (180, 100, 100),
]

def fetch_snapshot(url: str) -> np.ndarray:
    print(f"[snapshot] fetching {url} ...")
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode snapshot JPEG")
    print(f"[snapshot] got frame {frame.shape[1]}x{frame.shape[0]}")
    return frame


def draw_detections(frame: np.ndarray, results, model) -> tuple[np.ndarray, int]:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    kept = 0

    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls)
            cls_name = model.names[cls_id]
            conf     = float(box.conf)

            if CLASS_FILTER and cls_name.lower() not in CLASS_FILTER:
                continue
            if conf < CONF_THRESHOLD:
                continue

            kept += 1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            colour = PALETTE[cls_id % len(PALETTE)]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # Centre crosshair
            cv2.drawMarker(annotated, (cx, cy), colour,
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

            # Label pill
            label = f"{cls_name}  {conf:.0%}"
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            (tw, text_h), baseline = cv2.getTextSize(label, font, fs, th)
            label_y = max(y1 - 6, text_h + 4)
            cv2.rectangle(annotated,
                          (x1, label_y - text_h - 4),
                          (x1 + tw + 6, label_y + baseline),
                          colour, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 3, label_y - 2),
                        font, fs, (0, 0, 0), th, cv2.LINE_AA)

            # Pixel coordinate annotation
            coord_text = f"({cx}, {cy})"
            cv2.putText(annotated, coord_text, (cx + 8, cy - 8),
                        font, 0.45, colour, 1, cv2.LINE_AA)

    # Top bar summary
    filter_str = ", ".join(CLASS_FILTER) if CLASS_FILTER else "all classes"
    summary = (
        f"model: {os.path.basename(MODEL_PATH)}"
        f"  |  conf >= {CONF_THRESHOLD:.0%}"
        f"  |  filter: {filter_str}"
        f"  |  detections: {kept}"
    )
    cv2.rectangle(annotated, (0, 0), (w, 26), (30, 30, 30), cv2.FILLED)
    cv2.putText(annotated, summary, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated, kept


def main():
    # Load model
    print(f"[yolo] loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[yolo] {len(model.names)} classes:")
    for idx, name in model.names.items():
        print(f"         {idx:2d}: {name}")

    if CLASS_FILTER:
        unknown = [c for c in CLASS_FILTER if c not in [n.lower() for n in model.names.values()]]
        if unknown:
            print(f"[warn] unknown CLASS_FILTER entries: {unknown}", file=sys.stderr)

    # Fetch live frame
    try:
        frame = fetch_snapshot(SNAPSHOT_URL)
    except Exception as e:
        print(f"[error] could not fetch snapshot: {e}", file=sys.stderr)
        sys.exit(1)

    # Run inference
    print(f"[yolo] running inference (conf >= {CONF_THRESHOLD}) ...")
    results = model(frame, verbose=False)

    # Print detections to stdout
    print("\n── Raw detections ────────────────────────────────────────────")
    total = 0
    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls)
            cls_name = model.names[cls_id]
            conf     = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            passes_filter = (not CLASS_FILTER or cls_name.lower() in CLASS_FILTER)
            passes_conf   = conf >= CONF_THRESHOLD
            mark = "✓" if (passes_filter and passes_conf) else "✗"
            print(f"  {mark} {cls_name:20s}  conf={conf:.2f}"
                  f"  bbox=[{x1},{y1},{x2},{y2}]  center=({cx},{cy})")
            total += 1
    print(f"── Total raw: {total}  (✓ = passes filter) ──────────────────\n")

    # Draw boxes and save
    annotated, kept = draw_detections(frame, results, model)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"[output] {kept} box(es) drawn → saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
