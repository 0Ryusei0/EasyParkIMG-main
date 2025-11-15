import cv2
from ultralytics import YOLO
import time
import threading
import numpy as np

url = "http://192.168.1.11:4747/video" 
model = YOLO("yolov8n.pt").to("cuda")

# Warmup GPU dengan format yang benar
print("🔥 Warming up GPU...")
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(source=dummy_frame, imgsz=640, verbose=False)
print("✓ GPU ready!\n")

class FrameReader:
    """Thread terpisah untuk baca frame - mencegah blocking"""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            time.sleep(0.01)  # Prevent CPU spike
                
    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.stopped = True
        self.cap.release()

# Inisialisasi
reader = FrameReader(url).start()
time.sleep(2)  # Tunggu buffer pertama

fps_time = time.time()
fps_counter = 0

print("🚀 Tekan ESC untuk keluar\n")

while True:
    ret, frame = reader.read()
    
    if not ret or frame is None:
        print("⚠ Frame hilang, skip...")
        time.sleep(0.1)
        continue
    
    # Resize untuk proses lebih cepat (opsional)
    frame = cv2.resize(frame, (640, 480))
    
    # Deteksi dengan conf threshold lebih tinggi (kurangi objek yang diproses)
    results = model(frame, conf=0.5, verbose=False)
    
    # Hitung FPS
    fps_counter += 1
    if time.time() - fps_time > 1:
        fps = fps_counter / (time.time() - fps_time)
        print(f"📊 FPS: {fps:.1f}")
        fps_counter = 0
        fps_time = time.time()
    
    # Tampilkan hasil
    annotated = results[0].plot()
    cv2.imshow("YOLOv8 + ESP32-CAM", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

reader.stop()
cv2.destroyAllWindows()