import sys
import os
import time
import cv2
import json
import requests
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from threading import Thread, Lock
from queue import Queue

# ============== PATH SETUP ==============
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ============== CONFIG ==============
STREAM_URL = "http://192.168.1.11:4747/video"
MODEL_PATH = r"D:\EasyParkIMG-main\Model\parking_detection2\weights\best.pt"
SLOT_MAP_JSON = ROOT / "slot_map.json"
LARAVEL_API = "http://localhost:8000/api/parking-slots/update"

# ROI Polygon
ROI_POLY = np.array([
    [120, 90],
    [520, 80],
    [560, 460],
    [100, 470],
], dtype=np.int32)

# Class names
CLASS_NAMES = {0: "terisi", 1: "kosong"}

# ============== GPU OPTIMIZATION ==============
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda:0' if USE_GPU else 'cpu'
print(f"\n{'='*60}")
print(f"🎮 GPU STATUS")
print(f"{'='*60}")
if USE_GPU:
    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device: {DEVICE}")
else:
    print(f"⚠️  GPU not available, using CPU")
print(f"{'='*60}\n")

# ============== OPTIMIZATIONS ==============
INFERENCE_SKIP = 2 if USE_GPU else 3  # GPU bisa process lebih sering
INFERENCE_SIZE = 416  # Keep original size for accuracy with GPU
USE_FP16 = USE_GPU  # Half precision untuk GPU

# Async API
api_queue = Queue(maxsize=2)
api_thread = None

# Shared state
status_lock = Lock()
current_slot_status = {}
current_summary = {"total": 0, "available": 0, "occupied": 0, "unknown": 0}

# ============== CORE FUNCTIONS ==============

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def map_detection_to_slot(detection_box, slot_zones, min_iou=0.25):
    """Map detection box ke slot ID berdasarkan IoU tertinggi"""
    best_slot = None
    best_iou = min_iou
    
    for slot_id, zone in slot_zones.items():
        slot_box = (
            zone["x"],
            zone["y"],
            zone["x"] + zone["w"],
            zone["y"] + zone["h"]
        )
        iou = calculate_iou(detection_box, slot_box)
        if iou > best_iou:
            best_iou = iou
            best_slot = slot_id
    
    return best_slot, best_iou


def process_detections(yolo_results, slot_zones, class_names, 
                      conf_thresh=0.5, min_area=400, min_iou=0.25):
    """Process YOLO detections dan build slot status"""
    slot_status = {slot_id: "unknown" for slot_id in slot_zones.keys()}
    slot_details = {}
    slot_best_conf = {slot_id: 0.0 for slot_id in slot_zones.keys()}

    if not hasattr(yolo_results, 'boxes') or len(yolo_results.boxes) == 0:
        summary = {
            "total": len(slot_zones),
            "available": 0,
            "occupied": 0,
            "unknown": len(slot_zones)
        }
        return {"slots": slot_status, "details": slot_details, "summary": summary}

    for box in yolo_results.boxes:
        try:
            xyxy = box.xyxy[0].cpu().numpy()
        except Exception:
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        try:
            class_id = int(box.cls[0])
        except Exception:
            class_id = int(box.cls)
        try:
            conf = float(box.conf[0])
        except Exception:
            conf = float(box.conf)

        if conf < conf_thresh:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        det_box = (int(x1), int(y1), int(x2), int(y2))
        slot_id, iou = map_detection_to_slot(det_box, slot_zones, min_iou=min_iou)

        if slot_id is None:
            continue

        if conf > slot_best_conf[slot_id]:
            status = class_names.get(class_id, "unknown")
            slot_status[slot_id] = status
            slot_best_conf[slot_id] = conf
            slot_details[slot_id] = {
                "status": status,
                "confidence": round(conf, 3),
                "iou": round(iou, 3),
                "box": det_box
            }

    available = sum(1 for s in slot_status.values() if s == "kosong")
    occupied = sum(1 for s in slot_status.values() if s == "terisi")
    unknown = sum(1 for s in slot_status.values() if s == "unknown")

    return {
        "slots": slot_status,
        "details": slot_details,
        "summary": {
            "total": len(slot_zones),
            "available": available,
            "occupied": occupied,
            "unknown": unknown
        }
    }


def crop_by_roi(frame, poly):
    """Crop frame by ROI polygon"""
    h, w = frame.shape[:2]
    if poly is None or len(poly) < 3:
        return frame, (0, 0)
    
    x, y, ww, hh = cv2.boundingRect(poly)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    crop = cv2.bitwise_and(frame, frame, mask=mask)[y:y+hh, x:x+ww]
    return crop, (x, y)


def draw_slot_zones(frame, slot_zones, slot_status, roi_offset=(0, 0)):
    """Draw slot zones dengan warna sesuai status"""
    ox, oy = roi_offset
    
    for slot_id, zone in slot_zones.items():
        x1 = int(zone["x"] + ox)
        y1 = int(zone["y"] + oy)
        x2 = int(x1 + zone["w"])
        y2 = int(y1 + zone["h"])
        
        status = slot_status.get(slot_id, "unknown")
        if status == "kosong":
            color = (0, 255, 0)      # Hijau
        elif status == "terisi":
            color = (0, 0, 255)      # Merah
        else:
            color = (128, 128, 128)  # Abu-abu
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{slot_id}: {status}"
        cv2.putText(frame, label, (x1 + 5, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


def api_worker():
    """Background thread untuk kirim API - NON-BLOCKING"""
    global api_queue
    
    while True:
        try:
            slot_status = api_queue.get(timeout=1)
            
            if slot_status is None:  # Poison pill
                break
            
            payload = {"slots": slot_status}
            headers = {"Content-Type": "application/json"}
            
            try:
                response = requests.post(
                    LARAVEL_API,
                    json=payload,
                    headers=headers,
                    timeout=3.0
                )
                
                if response.status_code == 200:
                    print(f"✅ API sent (200) at {time.strftime('%H:%M:%S')}")
                else:
                    print(f"⚠ API response: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ API error: {e}")
            
            api_queue.task_done()
            
        except Exception:
            continue


def send_to_laravel_async(slot_status):
    """Queue API request (non-blocking)"""
    try:
        if not api_queue.full():
            api_queue.put_nowait(slot_status.copy())
    except Exception:
        pass


def load_slot_map(path):
    """Load slot map dari JSON"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Slot map not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calibrate_slots_interactive(video_source, roi_poly, save_path):
    """
    Tool kalibrasi interaktif untuk mendapatkan koordinat slot
    
    Cara pakai:
    1. Klik kiri-atas slot
    2. Klik kanan-bawah slot
    3. Press 's' untuk save
    4. Press 'q' untuk quit
    """
    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Tidak bisa baca frame dari camera")
        cap.release()
        return

    # Resize to match detection resolution
    frame = cv2.resize(frame, (640, 480))
    
    # Crop ROI
    crop, (ox, oy) = crop_by_roi(frame, roi_poly)
    cv2.imwrite("calibration_crop.jpg", crop)
    
    print("\n" + "="*60)
    print("📍 SLOT CALIBRATION TOOL")
    print("="*60)
    print(f"✅ Screenshot saved: calibration_crop.jpg")
    print(f"   Ukuran crop: {crop.shape[1]}x{crop.shape[0]} (w x h)\n")
    print("Instruksi:")
    print("  1. Klik kiri-atas slot → klik kanan-bawah slot")
    print("  2. Ulangi untuk semua slot (A1-A6, B1-B6)")
    print("  3. Press 's' untuk save")
    print("  4. Press 'q' untuk selesai")
    print("="*60 + "\n")

    slots = {}
    points = []
    slot_names = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6"]
    current_idx = 0

    def mouse_cb(event, mx, my, flags, param):
        nonlocal points, slots, current_idx, crop
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((mx, my))
            cv2.circle(crop, (mx, my), 4, (0, 255, 0), -1)
            cv2.imshow("Calibration", crop)
            
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                x_slot = int(min(x1, x2))
                y_slot = int(min(y1, y2))
                w_slot = int(abs(x2 - x1))
                h_slot = int(abs(y2 - y1))
                
                if current_idx < len(slot_names):
                    slot_id = slot_names[current_idx]
                    slots[slot_id] = {
                        "x": x_slot,
                        "y": y_slot,
                        "w": w_slot,
                        "h": h_slot
                    }
                    
                    cv2.rectangle(crop, (x_slot, y_slot),
                                (x_slot + w_slot, y_slot + h_slot),
                                (0, 255, 255), 2)
                    
                    cv2.putText(crop, slot_id, (x_slot + 5, y_slot + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    print(f"✅ {slot_id}: x={x_slot}, y={y_slot}, w={w_slot}, h={h_slot}")
                    current_idx += 1
                    
                    if current_idx >= len(slot_names):
                        print("\n🎉 Semua slot sudah dikalibrasi!")
                        print("   Press 's' untuk save atau 'q' untuk quit")
                
                points = []
                cv2.imshow("Calibration", crop)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_cb)
    cv2.imshow("Calibration", crop)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(slots, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Slot map saved to: {save_path}")
        
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    if slots:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(slots, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Final slot map saved to: {save_path}")


# ============== MAIN ==============

def main():
    global api_thread, current_slot_status, current_summary
    
    print("\n" + "="*60)
    print("🚗 GPU-ACCELERATED PARKING DETECTION SYSTEM")
    print("="*60)
    
    # Check if slot_map.json exists
    if not SLOT_MAP_JSON.exists():
        print(f"\n⚠  Slot map tidak ditemukan: {SLOT_MAP_JSON}")
        print("Jalankan kalibrasi? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            print("\n🔧 Memulai kalibrasi...")
            calibrate_slots_interactive(STREAM_URL, ROI_POLY, SLOT_MAP_JSON)
            print("\n✅ Kalibrasi selesai!")
            print("   Restart script untuk memulai detection.\n")
            return
        else:
            print("❌ Tidak bisa lanjut tanpa slot map. Exit.")
            return
    
    # Load slot map
    print(f"📂 Loading slot map: {SLOT_MAP_JSON}")
    try:
        slot_zones = load_slot_map(SLOT_MAP_JSON)
    except Exception as e:
        print("❌ Gagal load slot map:", e)
        return

    print(f"✅ Loaded {len(slot_zones)} slots: {list(slot_zones.keys())}\n")
    
    # Load model with GPU
    print("🔥 Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        
        # Warm up GPU
        if USE_GPU:
            print("🔥 Warming up GPU...")
            dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
            for _ in range(3):  # Multiple warm-up runs
                _ = model.predict(
                    dummy, 
                    verbose=False, 
                    imgsz=INFERENCE_SIZE,
                    half=USE_FP16,
                    device=DEVICE
                )
            print("✅ GPU warmed up!")
        
    except Exception as e:
        print("❌ Gagal load model YOLO:", e)
        return
    print("✅ Model loaded\n")
    
    # Start API worker thread
    api_thread = Thread(target=api_worker, daemon=True)
    api_thread.start()
    print("✅ API worker thread started\n")
    
    # Camera
    print(f"📸 Connecting to: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)
    
    # Camera optimizations
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    time.sleep(1.0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera stream")
        return
    
    print("✅ Camera connected\n")
    print("="*60)
    print("⚡ OPTIMIZATIONS:")
    print(f"   - Device: {DEVICE}")
    print(f"   - Skip frames: 1/{INFERENCE_SKIP}")
    print(f"   - Inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE}")
    print(f"   - FP16 (Half precision): {USE_FP16}")
    print(f"   - Async API: Enabled")
    print("="*60)
    print("🎬 Starting detection loop...")
    print("   Press ESC to quit")
    print("="*60 + "\n")
    
    # Main loop
    frame_count = 0
    send_interval = 2.0
    last_send_time = time.time()
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    ox, oy = 0, 0
    
    # Initialize status
    with status_lock:
        current_slot_status = {slot_id: "unknown" for slot_id in slot_zones.keys()}
        current_summary = {"total": len(slot_zones), "available": 0, "occupied": 0, "unknown": len(slot_zones)}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        fps_counter += 1
        
        # FPS calculation
        if time.time() - fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Resize to expected resolution (SAME AS CALIBRATION)
        frame = cv2.resize(frame, (640, 480))
        
        # INFERENCE - Only every N frames
        if frame_count % INFERENCE_SKIP == 0:
            # Crop ROI
            crop, (ox, oy) = crop_by_roi(frame, ROI_POLY)
            
            # YOLO inference with GPU
            try:
                results = model.predict(
                    crop,
                    conf=0.30,
                    iou=0.5,
                    imgsz=INFERENCE_SIZE,
                    verbose=False,
                    half=USE_FP16,
                    device=DEVICE
                )[0]
                
                # Process
                status_data = process_detections(
                    results,
                    slot_zones,
                    CLASS_NAMES,
                    conf_thresh=0.5,
                    min_area=400,
                    min_iou=0.25
                )
                
                # Update shared state
                with status_lock:
                    current_slot_status = status_data["slots"]
                    current_summary = status_data["summary"]
                
            except Exception as e:
                print(f"⚠️ Inference error: {e}")
        
        # Get current state (thread-safe)
        with status_lock:
            display_status = current_slot_status.copy()
            display_summary = current_summary.copy()
        
        # Send to API (async)
        current_time = time.time()
        if current_time - last_send_time >= send_interval:
            send_to_laravel_async(display_status)
            last_send_time = current_time
        
        # Print summary every 30 frames
        if frame_count % 30 == 0:
            print(f"\n📊 Frame {frame_count} | FPS: {current_fps}")
            print(f"   ✅ Available: {display_summary['available']}")
            print(f"   ❌ Occupied: {display_summary['occupied']}")
            print(f"   ❓ Unknown: {display_summary['unknown']}")
        
        # Visualization
        vis = draw_slot_zones(frame, slot_zones, display_status, roi_offset=(ox, oy))
        
        # Info overlay
        cv2.putText(vis, f"FPS: {current_fps} | GPU: {USE_GPU}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, f"Available: {display_summary['available']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Occupied: {display_summary['occupied']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Parking Detection (GPU)", vis)
        
        # Quit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    print("\n🛑 Stopping...")
    
    # Stop API thread
    api_queue.put(None)
    if api_thread:
        api_thread.join(timeout=2)
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!\n")


if __name__ == "__main__":
    main()