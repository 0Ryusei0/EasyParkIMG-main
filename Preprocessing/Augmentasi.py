# =========================================================
# 🚗 Parking Lot Dataset Augmentation Script
# Tujuan: Membuat variasi gambar (bayangan, gelap, noise)
# untuk melatih model YOLO agar lebih robust di kondisi nyata
# =========================================================

import cv2            # OpenCV untuk manipulasi gambar
import numpy as np     # Numpy untuk operasi numerik (matriks)
from pathlib import Path   # Untuk manajemen path file/folder yang cross-platform
import random          # Untuk membuat efek acak (bayangan, noise, dsb)
import shutil          # Untuk menyalin file (gambar dan label)

# =========================================================
# 🔹 Fungsi: apply_shadow()
# Menambahkan bayangan acak pada gambar untuk simulasi kondisi real
# =========================================================
def apply_shadow(image):
    """Tambahkan bayangan acak pada gambar"""
    h, w = image.shape[:2]  # Ambil tinggi dan lebar gambar
    
    # Pilih jenis bayangan secara acak
    shadow_type = random.choice(['vertical', 'horizontal', 'diagonal', 'random'])
    
    # Bayangan vertikal seperti tiang atau pohon
    if shadow_type == 'vertical':
        shadow_width = random.randint(w//8, w//3)   # Lebar bayangan
        shadow_pos = random.randint(0, w - shadow_width)  # Posisi mulai bayangan
        mask = np.ones_like(image, dtype=np.float32)
        mask[:, shadow_pos:shadow_pos+shadow_width] = random.uniform(0.3, 0.6)  # Gelapkan area
    
    # Bayangan horizontal seperti atap
    elif shadow_type == 'horizontal':
        shadow_height = random.randint(h//8, h//2)
        shadow_pos = random.randint(0, h - shadow_height)
        mask = np.ones_like(image, dtype=np.float32)
        mask[shadow_pos:shadow_pos+shadow_height, :] = random.uniform(0.3, 0.6)
    
    # Bayangan diagonal seperti dari arah miring
    elif shadow_type == 'diagonal':
        mask = np.ones_like(image, dtype=np.float32)
        # Buat poligon acak diagonal
        pts = np.array([
            [random.randint(0, w//2), 0],
            [random.randint(w//2, w), 0],
            [random.randint(w//2, w), h],
            [random.randint(0, w//2), h]
        ])
        cv2.fillPoly(mask, [pts], (random.uniform(0.3, 0.6),) * 3)
    
    # Bayangan acak (irregular) seperti awan atau bayangan orang
    else:
        mask = np.ones_like(image, dtype=np.float32)
        num_shadows = random.randint(1, 3)  # Jumlah bayangan acak
        for _ in range(num_shadows):
            x = random.randint(0, w)
            y = random.randint(0, h)
            radius = random.randint(min(h, w)//8, min(h, w)//3)
            cv2.circle(mask, (x, y), radius, 
                      (random.uniform(0.3, 0.6),) * 3, -1)  # Tambahkan area gelap
    
    # Blur agar transisi bayangan halus (tidak kaku)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Terapkan mask ke gambar (bayangan = hasil kali)
    result = (image * mask).astype(np.uint8)
    
    return result

# =========================================================
# 🔹 Fungsi: adjust_brightness()
# Mengubah tingkat kecerahan gambar untuk simulasi siang/malam
# =========================================================
def adjust_brightness(image, factor=None):
    """Ubah brightness gambar"""
    if factor is None:
        factor = random.uniform(0.4, 1.5)  # 0.4 = gelap, 1.5 = terang
    
    # Ubah ke HSV agar mudah ubah "Value" (kecerahan)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result

# =========================================================
# 🔹 Fungsi: add_noise()
# Menambahkan noise untuk meniru kondisi kamera gelap atau buram
# =========================================================
def add_noise(image):
    """Tambahkan noise untuk simulasi low-light"""
    noise = np.random.normal(0, random.randint(10, 25), image.shape)  # Gaussian noise
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

# =========================================================
# 🔹 Fungsi utama: augment_dataset()
# Men-generate dataset baru dengan variasi kondisi realistik
# =========================================================
def augment_dataset(source_dir, target_dir, num_variations=3):
    """
    Augmentasi dataset untuk kondisi gelap/bayangan
    
    Args:
        source_dir: Folder dataset asli (harus ada train/images & train/labels)
        target_dir: Folder hasil augmentasi
        num_variations: Jumlah variasi augmentasi per gambar
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Buat struktur folder (train & val)
    for split in ['train', 'val']:
        (target_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Loop untuk dua folder: train dan val
    for split in ['train', 'val']:
        image_dir = source_dir / split / 'images'
        label_dir = source_dir / split / 'labels'
        
        # Kalau folder tidak ada → skip
        if not image_dir.exists():
            print(f"Warning: {image_dir} tidak ditemukan!")
            continue
        
        # Ambil semua file gambar (jpg/png)
        image_files = list(image_dir.glob('*.[jp][pn]g'))
        print(f"\nMemproses {len(image_files)} gambar dari {split}...")
        
        # Proses tiap gambar satu per satu
        for img_path in image_files:
            # Copy gambar asli dulu ke folder baru
            shutil.copy(img_path, target_dir / split / 'images' / img_path.name)
            
            # Copy label (jika ada)
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, target_dir / split / 'labels' / label_path.name)
            
            # Load gambar pakai OpenCV
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Buat beberapa variasi augmentasi
            for i in range(num_variations):
                # Pilih tipe augmentasi secara acak
                aug_type = random.choice([
                    'shadow_only',
                    'dark_only', 
                    'shadow_dark',
                    'dark_noise',
                    'shadow_dark_noise'
                ])
                
                aug_image = image.copy()
                
                # Terapkan bayangan jika tipe-nya mengandung "shadow"
                if 'shadow' in aug_type:
                    aug_image = apply_shadow(aug_image)
                
                # Ubah brightness jika tipe-nya mengandung "dark"
                if 'dark' in aug_type:
                    aug_image = adjust_brightness(aug_image, 
                                                 random.uniform(0.4, 0.8))
                
                # Tambahkan noise jika tipe-nya mengandung "noise"
                if 'noise' in aug_type:
                    aug_image = add_noise(aug_image)
                
                # Simpan hasil augmentasi ke folder output
                new_name = f"{img_path.stem}_aug{i}_{aug_type}{img_path.suffix}"
                cv2.imwrite(str(target_dir / split / 'images' / new_name), aug_image)
                
                # Copy file label dengan nama yang sama
                if label_path.exists():
                    new_label = f"{img_path.stem}_aug{i}_{aug_type}.txt"
                    shutil.copy(label_path, target_dir / split / 'labels' / new_label)
        
        print(f"Selesai memproses {split}!")
    
    # Notifikasi akhir
    print("\n" + "="*50)
    print("✅ Augmentasi selesai!")
    print(f"📁 Dataset baru tersimpan di: {target_dir}")
    print("="*50)

# =========================================================
# 🔹 Eksekusi utama
# =========================================================
if __name__ == '__main__':
    # Tentukan lokasi dataset sumber dan target output
    source_dataset = "D:\\Quant_ML_Project\\ML.py\\EasyPark\\Dataset\\parking_lot_final"
    target_dataset = "D:\\Quant_ML_Project\\ML.py\\EasyPark\\Dataset\\parking_lot_aug"
    
    # Jalankan augmentasi (3 variasi per gambar)
    augment_dataset(
        source_dir=source_dataset,
        target_dir=target_dataset,
        num_variations=3
    )
    
    # Pengingat agar update path YAML YOLO ke dataset baru
    print("\n🟡 Jangan lupa update data.yaml ke dataset_augmented!")
