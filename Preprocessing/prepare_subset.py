import os
import shutil
import random

# === Path asal (pakai path relatif dari file script ini) ===
BASE_DIR = os.path.dirname(__file__)  # folder tempat file prepare_subset.py berada
SRC_BASE = os.path.join(BASE_DIR, "Dataset", "parking_lot")
SRC_IMAGES = os.path.join(SRC_BASE, "train", "images")
SRC_LABELS = os.path.join(SRC_BASE, "train", "labels")

# === Path tujuan (subset) ===
DST_BASE = os.path.join(BASE_DIR, "Dataset", "parking_lot_subset")
SPLITS = {
    "train": 50,
    "valid": 10,
    "test": 10
}

# === Buat folder subset ===
for split in SPLITS.keys():
    os.makedirs(os.path.join(DST_BASE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST_BASE, split, "labels"), exist_ok=True)

# === Ambil semua file image ===
all_images = [f for f in os.listdir(SRC_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_images)

# === Bagi sesuai split ===
start = 0
for split, count in SPLITS.items():
    subset_images = all_images[start:start + count]
    start += count

    for img_name in subset_images:
        label_name = os.path.splitext(img_name)[0] + ".txt"

        src_img = os.path.join(SRC_IMAGES, img_name)
        src_lbl = os.path.join(SRC_LABELS, label_name)
        dst_img = os.path.join(DST_BASE, split, "images", img_name)
        dst_lbl = os.path.join(DST_BASE, split, "labels", label_name)

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

    print(f"✅ {split} set selesai: {len(subset_images)} file disalin.")

print("\n🎉 Semua subset selesai dibuat di folder 'Dataset/parking_lot_subset/'")
