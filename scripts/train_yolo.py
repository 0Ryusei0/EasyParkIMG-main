# =====================================================
# 📦 IMPORT LIBRARY
# =====================================================
from ultralytics import YOLO   # Import class YOLO dari library ultralytics (untuk deteksi & training)
import torch                   # Library PyTorch, digunakan YOLO untuk komputasi GPU/CPU


# =====================================================
# 🚀 MAIN PROGRAM
# =====================================================
if __name__ == '__main__':     # Pastikan kode hanya berjalan jika file ini dijalankan langsung (bukan di-import)
    
    # 🧩 Set multiprocessing method (khusus Windows)
    # 'spawn' digunakan agar proses parallel PyTorch bisa berjalan tanpa error di OS Windows
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # =================================================
    # 📂 LOAD MODEL
    # =================================================
    model = YOLO('yolov8n.pt')   # Memuat model YOLOv8 versi nano (model pre-trained kecil & cepat)
    
    
    # =================================================
    # 🏋️‍♂️ TRAINING MODEL
    # =================================================
    results = model.train(
        data='D:\Quant_ML_Project\ML.py\EasyPark\configs\data.yaml',  # Path ke konfigurasi dataset (train/val)
        epochs=100,            # Jumlah total iterasi training
        imgsz=640,             # Ukuran gambar input yang akan digunakan YOLO
        batch=8,               # Jumlah gambar per batch (semakin besar, semakin cepat tapi butuh RAM besar)
        name='parking_detection',  # Nama folder hasil training di dalam runs/train/
        patience=50,           # Early stopping: hentikan training jika tidak ada peningkatan 50 epoch
        save=True,             # Simpan hasil model terbaik (.pt)
        device=0,              # Gunakan GPU index ke-0 (atau 'cpu' kalau tidak punya GPU)
        plots=True,            # Simpan grafik loss, mAP, precision, recall, dll.
        workers=0              # Disable multiprocessing worker (mencegah bug di Windows)
    )
    
    
    # =================================================
    # 📊 HASIL TRAINING
    # =================================================
    print(results)  # Menampilkan hasil ringkasan training (misal mAP, loss, waktu, dsb.)
