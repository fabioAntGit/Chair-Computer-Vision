import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "runs/detect/runs/train/foe-bot-exp12/weights/best.pt"
INPUT_DIR    = Path("input")
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ─────────────────────────────────────────────────────────────────────────────
# Load the trained model
# ─────────────────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Run inference on each image
# ─────────────────────────────────────────────────────────────────────────────
image_paths = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in SUPPORTED_EXTENSIONS
]

if not image_paths:
    print(f"No images found in '{INPUT_DIR}'. Supported formats: {SUPPORTED_EXTENSIONS}")

for image_path in image_paths:
    print(f"Processing: {image_path.name}")

    # ── Run prediction ────────────────────────────────────────────────────────
    results = model.predict(
        source=str(image_path),
        conf=0.25,          # Minimum confidence threshold to consider a detection.
                            # Detections below this value are discarded.
                            # Range: 0.0–1.0. Lower = more detections, more noise.

        iou=0.7,            # IoU threshold for Non-Maximum Suppression (NMS).
                            # Overlapping boxes above this threshold are merged.
                            # Lower = stricter deduplication.

        imgsz=640,          # Resize input to this size before inference.
                            # Should match the size used during training.

        max_det=300,        # Maximum number of detections per image.

        device="mps",       # Inference device. Use "cpu", 0, [0,1], or "mps".

        verbose=False,      # Suppress per-image console output.
    )

    result = results[0]  # One result per image

    # ── Build the detections JSON ─────────────────────────────────────────────
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()   # Bounding box corners (pixels)
        detections.append({
            "class_id":   int(box.cls),                    # Numeric class index
            "class_name": model.names[int(box.cls)],       # Human-readable class label
            "confidence": round(float(box.conf), 4),       # Detection confidence score
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width":  round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            }
        })

    # ── Save JSON file ────────────────────────────────────────────────────────
    json_path = OUTPUT_DIR / f"{image_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)

    # ── Save annotated image ──────────────────────────────────────────────────
    # result.plot() returns a BGR numpy array with bounding boxes drawn on it
    annotated_image = result.plot(
        conf=True,      # Show confidence score on each label
        labels=True,    # Show class name on each box
        boxes=True,     # Draw bounding boxes
        line_width=2,   # Box border thickness in pixels
    )
    output_image_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_image_path), annotated_image)

    print(f"  → {len(detections)} detection(s) | image saved to '{output_image_path}' | JSON saved to '{json_path}'")

print("\nDone.")