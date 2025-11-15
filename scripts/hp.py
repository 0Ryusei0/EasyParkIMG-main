# ===============================================================
# 🧠 SISTEM DETEKSI PARKIR BERBASIS YOLOv8 + FLASK STREAMING
# ===============================================================

import cv2                           # Library untuk pengolahan citra dan video
from ultralytics import YOLO         # Library YOLOv8 untuk deteksi objek
import time                          # Untuk menghitung FPS (frame per second)
import os                            # Untuk pengecekan file model
from flask import Flask, Response    # Flask untuk web server dan streaming

# ======================
# 🔧 INISIALISASI FLASK
# ======================
app = Flask(__name__)                # Membuat instance aplikasi Flask

# ======================
# ⚙️ KONFIGURASI DASAR
# ======================
url = "http://192.168.1.11:4747/video"   # URL stream dari kamera HP (IP Webcam)
model_path = r"D:\Quant_ML_Project\ML.py\EasyPark\Model\parking_detection2\weights\best.pt"

# Pastikan model YOLO tersedia di path yang ditentukan
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model tidak ditemukan: {model_path}")

# ======================
# 🚀 LOAD MODEL YOLO
# ======================
print("🔥 Loading model...")
try:
    # Coba load model ke GPU (CUDA)
    model = YOLO(model_path).to("cuda")
    print("✅ Model di GPU\n")
except:
    # Jika tidak ada GPU, gunakan CPU
    model = YOLO(model_path)
    print("✅ Model di CPU\n")

# ======================
# 🎥 KONEKSI KAMERA
# ======================
print("📸 Connecting to camera...")
cap = cv2.VideoCapture(url)  # Membuka koneksi video stream dari IP camera

if not cap.isOpened():       # Jika gagal membuka kamera
    raise Exception("❌ Cannot connect to camera")

# Set resolusi video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ======================
# 🎬 GENERATOR FRAME STREAM
# ======================
def generate():
    prev_time = time.time()     # Waktu awal untuk perhitungan FPS
    fps_counter = 0             # Hitungan frame sementara
    fps_display = 0             # FPS yang akan ditampilkan

    while True:
        ret, frame = cap.read()  # Baca satu frame dari kamera
        if not ret:              # Jika gagal baca frame, tunggu sebentar
            time.sleep(0.1)
            continue
        
        # ======================
        # 🔍 DETEKSI OBJEK YOLO
        # ======================
        results = model(frame, conf=0.5, verbose=False)  # Jalankan deteksi YOLO
        annotated_frame = results[0].plot()              # Gambar bounding box ke frame

        # ======================
        # ⏱️ HITUNG FPS
        # ======================
        fps_counter += 1
        current_time = time.time()
        if current_time - prev_time >= 1:                # Update tiap 1 detik
            fps_display = fps_counter / (current_time - prev_time)
            fps_counter = 0
            prev_time = current_time
        
        # ======================
        # 🧾 TAMBAHKAN INFO KE FRAME
        # ======================
        # Menampilkan FPS di kiri atas
        cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hitung jumlah objek yang terdeteksi
        num_detections = len(results[0].boxes)
        
        # Tampilkan jumlah objek
        cv2.putText(annotated_frame, f"Objects: {num_detections}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # ======================
        # 🧩 ENKODE KE JPEG
        # ======================
        # Konversi frame ke format JPEG agar bisa dikirim lewat HTTP
        ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()   # Ubah hasil encode ke bytes

        # ======================
        # 🌐 STREAM DATA FRAME
        # ======================
        # Kirim frame ke browser dengan format multipart (streaming berkelanjutan)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ======================
# 🌍 ROUTE FLASK - HALAMAN UTAMA
# ======================
@app.route('/')
def index():
    # HTML tampilan utama dengan gaya modern (UI streaming)
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parking Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
                color: #fff;
                font-family: 'Segoe UI', sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                min-height: 100vh;
            }
            h1 {
                color: #00ff88;
                text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
                margin-bottom: 10px;
                font-size: 2rem;
            }
            .status {
                background: linear-gradient(90deg, #ff0000, #ff4444);
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                margin-bottom: 20px;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            .container {
                position: relative;
                max-width: 95%;
                width: 100%;
                max-width: 1200px;
                box-shadow: 0 20px 60px rgba(0, 255, 136, 0.2);
                border-radius: 15px;
                overflow: hidden;
                border: 2px solid #00ff88;
            }
            img { width: 100%; display: block; }
            .info {
                margin-top: 30px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                max-width: 1200px;
                width: 100%;
            }
            .card {
                background: rgba(0, 255, 136, 0.1);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid rgba(0, 255, 136, 0.3);
                text-align: center;
            }
            .card-icon { font-size: 2rem; margin-bottom: 10px; }
            .card-title { color: #00ff88; font-weight: bold; margin-bottom: 5px; }
            .card-text { color: #aaa; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <h1>🚗 Parking Detection System</h1>
        <div class="status">🔴 LIVE</div>
        <div class="container">
            <img src="/video" alt="Stream">  <!-- Stream video dari route /video -->
        </div>
        <div class="info">
            <div class="card">
                <div class="card-icon">📡</div>
                <div class="card-title">Source</div>
                <div class="card-text">IP Webcam Stream</div>
            </div>
            <div class="card">
                <div class="card-icon">🎯</div>
                <div class="card-title">Model</div>
                <div class="card-text">YOLOv8 Nano</div>
            </div>
            <div class="card">
                <div class="card-icon">⚡</div>
                <div class="card-title">Acceleration</div>
                <div class="card-text">CUDA GPU</div>
            </div>
        </div>
    </body>
    </html>
    """

# ======================
# 🎥 ROUTE FLASK - STREAM VIDEO
# ======================
@app.route('/video')
def video():
    # Mengirim hasil deteksi secara real-time
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================
# 🚀 MENJALANKAN SERVER FLASK
# ======================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌐 Web Server Running!")
    print("="*50)
    print("🔗 Open in browser:")
    print("   http://localhost:5000")
    print("\n⌨️  Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    # Jalankan Flask di semua IP (agar bisa diakses dari HP/laptop lain dalam 1 jaringan)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
