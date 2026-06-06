from ultralytics import YOLO
import os

# ─── CONFIG ───────────────────────────────────────────────────────
YAML_PATH  = r"C:\Users\vinod\Desktop\face detection 2\yolo\dataset.yaml"
EPOCHS     = 30
IMG_SIZE   = 640
BATCH_SIZE = 8
DEVICE     = "cpu"

# ─── CHECK ────────────────────────────────────────────────────────
if not os.path.exists(YAML_PATH):
    print(f"[ERROR] dataset.yaml not found!")
    exit()

print("=" * 50)
print("  YOLOv8 Face Detection - Training")
print("=" * 50)
print(f"  YAML   : {YAML_PATH}")
print(f"  Epochs : {EPOCHS}")
print(f"  Device : {DEVICE}\n")

# ─── TRANSFER LEARNING ────────────────────────────────────────────
model = YOLO("yolov8n.pt")

results = model.train(
    data       = YAML_PATH,
    epochs     = EPOCHS,
    imgsz      = IMG_SIZE,
    batch      = BATCH_SIZE,
    device     = DEVICE,
    patience   = 10,
    pretrained = True,
    name       = "face_yolo",
    verbose    = True,
)

# ─── RESULTS ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Training Complete!")
print("=" * 50)
print(f"  mAP@50   : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
print(f"  Recall   : {results.results_dict.get('metrics/recall(B)', 0):.4f}")
print(f"\n  Best model saved:")
print(f"  runs/detect/face_yolo/weights/best.pt")
print("=" * 50)