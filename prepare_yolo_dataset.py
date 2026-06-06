# prepare_yolo_dataset.py  (face detection 2 folder-හිදී save කරන්න)
import os, shutil, cv2, random

DATASET_PATH = r"C:\Users\vinod\Desktop\face detection 2\dataset"
OUTPUT_PATH  = r"C:\Users\vinod\Desktop\face detection 2\yolo_data"
CLASSES      = ["Hrithik Roshan", "Jessica Alba", "Robert Downey"]
TRAIN_SPLIT  = 0.8

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

for split in ['train', 'val']:
    os.makedirs(f"{OUTPUT_PATH}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/{split}/labels", exist_ok=True)

all_files = []
for class_name in CLASSES:
    folder = os.path.join(DATASET_PATH, class_name)
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            all_files.append(os.path.join(folder, f))

random.shuffle(all_files)
split_idx = int(len(all_files) * TRAIN_SPLIT)
splits = {'train': all_files[:split_idx], 'val': all_files[split_idx:]}

total_labeled = 0
for split_name, files in splits.items():
    for img_path in files:
        img = cv2.imread(img_path)
        if img is None: continue
        
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w  = img.shape[:2]
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        
        if len(faces) == 0: continue
        
        stem    = os.path.splitext(os.path.basename(img_path))[0]
        img_out = f"{OUTPUT_PATH}/{split_name}/images/{stem}.jpg"
        lbl_out = f"{OUTPUT_PATH}/{split_name}/labels/{stem}.txt"
        
        shutil.copy2(img_path, img_out)
        
        with open(lbl_out, 'w') as f:
            for (x, y, bw, bh) in faces:
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        total_labeled += 1

print(f"[OK] {total_labeled} images prepared")
print(f"     Train: {len(splits['train'])} | Val: {len(splits['val'])}")

# dataset.yaml update
yaml = f"""path: {OUTPUT_PATH}
train: train/images
val: val/images

nc: 1
names: ['face']
"""
yaml_path = r"C:\Users\vinod\Desktop\face detection 2\yolo\dataset.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml)
print(f"[OK] dataset.yaml updated")