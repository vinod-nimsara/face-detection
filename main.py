# main.py  (face detection 2 folder-හිදී)
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from datetime import datetime
import os

# ─── Load Models ──────────────────────────────────────────────────
print("[*] Loading models...")

YOLO_MODEL = r"yolo\runs\detect\face_yolo\weights\best.pt"
CNN_MODEL  = r"cnn\cnn_model.h5"
LABELS     = r"cnn\labels.txt"

# YOLO — face detector
yolo = YOLO(YOLO_MODEL)

# CNN — face recognizer  
cnn  = tf.keras.models.load_model(CNN_MODEL)

# Class names
with open(LABELS) as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"[OK] Classes: {class_names}\n")

# ─── Access Log ───────────────────────────────────────────────────
LOG_FILE = "logs/access_log.txt"
os.makedirs("logs", exist_ok=True)

def log_access(name, confidence, status):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {name} | Conf: {confidence:.1%} | {status}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(f"  [LOG] {line.strip()}")

# ─── Recognize Face (CNN) ─────────────────────────────────────────
def recognize_face(face_img):
    """Cropped face image → person name + confidence"""
    img = cv2.resize(face_img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 100, 100, 1)
    
    predictions = cnn.predict(img, verbose=0)
    idx         = np.argmax(predictions[0])
    confidence  = predictions[0][idx]
    name        = class_names[idx]
    return name, confidence

# ─── Main Security Loop ───────────────────────────────────────────
print("[*] Starting Security System...")
print("    Press 'Q' to quit | 'S' to screenshot\n")

CONF_THRESHOLD   = 0.60   # CNN confidence — මීට වඩා අඩු නම් Unknown
YOLO_CONF        = 0.40   # YOLO face detection confidence

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screenshot_count = 0
os.makedirs("screenshots", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── STEP 1: YOLO — Detect faces ──
    yolo_results = yolo(frame, conf=YOLO_CONF, verbose=False)

    for result in yolo_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounds check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # ── STEP 2: CNN — Recognize face ──
            name, conf = recognize_face(face_crop)

            if conf >= CONF_THRESHOLD:
                status = "AUTHORIZED"
                color  = (0, 220, 0)      # Green
            else:
                name   = "Unknown"
                status = "UNAUTHORIZED"
                color  = (0, 0, 255)      # Red

            # ── Draw bounding box ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({conf:.0%})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-lh-12), (x1+lw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Status badge
            cv2.putText(frame, status, (x1, y2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Log
            log_access(name, conf, status)

    # ── Timestamp overlay ──
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Security System | {ts}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Q: Quit  S: Screenshot", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("Face Detection & Recognition - Security System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        path = f"screenshots/capture_{screenshot_count:04d}.jpg"
        cv2.imwrite(path, frame)
        print(f"  [Screenshot] {path}")
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
print("\n[OK] Security system stopped.")
print(f"[OK] Access log: {LOG_FILE}")