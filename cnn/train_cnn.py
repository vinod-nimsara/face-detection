import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ─── CONFIG ───────────────────────────────────────────────────────
DATASET_PATH = "../dataset"   # cnn folder ඇතුළේ නිසා ../ use කරනවා
IMG_SIZE     = 100
CLASSES      = ["Hrithik Roshan", "Jessica Alba", "Robert Downey"]

print("=" * 50)
print("  CNN Face Recognition - Training")
print("=" * 50)

# ─── STEP 1: Data Load ────────────────────────────────────────────
X = []
y = []

for label_idx, class_name in enumerate(CLASSES):
    class_folder = os.path.join(DATASET_PATH, class_name)
    
    if not os.path.exists(class_folder):
        print(f"[ERROR] Folder not found: {class_folder}")
        continue
    
    images_found = 0
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_idx)
        images_found += 1
    
    print(f"  [{label_idx}] {class_name}: {images_found} images loaded")

# ─── STEP 2: Preprocess ───────────────────────────────────────────
X = np.array(X, dtype='float32') / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = to_categorical(np.array(y), num_classes=len(CLASSES))

print(f"\n  Total images : {X.shape[0]}")
print(f"  Image shape  : {X.shape[1:]}")

# ─── STEP 3: Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train set    : {X_train.shape[0]}")
print(f"  Test set     : {X_test.shape[0]}\n")

# ─── STEP 4: CNN Model ────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ─── STEP 5: Train ────────────────────────────────────────────────
print("\n[*] Training started...\n")
model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_test, y_test),
    batch_size=32
)

# ─── STEP 6: Evaluate + Save ──────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Accuracy : {acc*100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")

model.save("cnn_model.h5")
print("\n[OK] Model saved: cnn_model.h5")

# Labels file save (recognition-ට පසුව use කිරීමට)
with open("labels.txt", "w") as f:
    for name in CLASSES:
        f.write(name + "\n")
print("[OK] Labels saved: labels.txt")