# =====================================================
# 🚗 Parking Lot Preprocessing (Tanpa ROI, Tanpa Gaussian)
# =====================================================

import cv2
import os

# === Path Dataset ===
INPUT_DIR = r"D:\Quant_ML_Project\ML.py\EasyPark\Dataset\parking_lot_final\train\images"
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), "images_preprocessed")

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Inisialisasi CLAHE ===
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# === Fungsi Processing ===
def preprocess_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal membaca: {image_path}")
        return

    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Terapkan CLAHE untuk peningkatan kontras
    enhanced = clahe.apply(gray)

    # *Tanpa Gaussian Blur — langsung Resize*
    resized = cv2.resize(enhanced, (640, 640))

    # Simpan hasilnya
    cv2.imwrite(save_path, resized)

# === Proses semua gambar ===
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)
        preprocess_image(in_path, out_path)

print(f"\n✅ Selesai! Semua gambar tersimpan di: {OUTPUT_DIR}")
