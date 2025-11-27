import cv2
import json
import requests
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from threading import Thread
from queue import Queue
import time
from datetime import datetime

# ============== CONFIG ==============
STREAM_URL = "http://192.168.1.12:4747/video"
MODEL_PATH = r"D:\EasyParkIMG-main\Model\parking_detection2\weights\best.pt"
SLOT_MAP_JSON = Path(__file__).parent / "slot_map.json"
LARAVEL_API = "http://localhost:8000/api/parking-slots/update"
LARAVEL_GET_API = "http://localhost:8000/api/parking-slots"

# ROI Polygon
ROI_POLY = np.array([[120, 90], [520, 80], [560, 460], [100, 470]], dtype=np.int32)

# Class names
CLASS_NAMES = {0: "terisi", 1: "kosong"}

# Slot names
SLOT_NAMES = ["A1", "A2", "A3", "A4", "A5", "A6", 
              "B1", "B2", "B3", "B4", "B5", "B6"]

# Queue untuk async API
api_queue = Queue(maxsize=2)

# Cache untuk reserved slots
reserved_slots_cache = {}

# ============== FUNCTIONS ==============

def fetch_reserved_slots():
    """Ambil daftar slot yang reserved dari database"""
    global reserved_slots_cache
    try:
        response = requests.get(LARAVEL_GET_API, timeout=10.0)
        if response.status_code == 200:
            slots_data = response.json()
            reserved = {}
            
            for slot in slots_data:
                if slot.get('status') == 'reserved':
                    slot_code = slot.get('slot_code')
                    if slot_code:
                        reserved[slot_code] = {
                            'status': 'reserved',
                            'last_update': slot.get('last_update')
                        }
            
            reserved_slots_cache = reserved
            if reserved:
                print(f"🔒 RESERVED SLOTS: {list(reserved.keys())} | {time.strftime('%H:%M:%S')}")
            else:
                print(f"✅ No reserved slots | {time.strftime('%H:%M:%S')}")
            
            return reserved
    except requests.exceptions.Timeout:
        print(f"⚠️  Reserve check timeout - using cache | {time.strftime('%H:%M:%S')}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Cannot connect to API for reserve check | {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"⚠️  Reserve check error: {str(e)[:50]} | {time.strftime('%H:%M:%S')}")
    
    # Return cached data jika API gagal
    return reserved_slots_cache

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def process_detections(results, slot_zones, reserved_slots, min_iou=0.15):
    """
    Process YOLO results dengan instant override untuk reserved slots
    PERBAIKAN: IoU lebih rendah, priority detection lebih tinggi
    """
    slot_status = {sid: "unknown" for sid in slot_zones.keys()}
    slot_conf = {sid: 0.0 for sid in slot_zones.keys()}
    
    # Tandai reserved slots dari database
    for slot_id in reserved_slots.keys():
        if slot_id in slot_status:
            slot_status[slot_id] = "reserved"
            slot_conf[slot_id] = 0.95  # Set tinggi tapi bukan max
    
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return slot_status
    
    # Proses semua detections
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # PERBAIKAN 1: Turunkan threshold confidence dan size
        if conf < 0.3 or (x2-x1)*(y2-y1) < 200:  # Lebih permisif
            continue
        
        det_box = (int(x1), int(y1), int(x2), int(y2))
        
        # Find best matching slot
        best_slot, best_iou = None, min_iou
        for sid, zone in slot_zones.items():
            slot_box = (zone["x"], zone["y"], 
                       zone["x"] + zone["w"], zone["y"] + zone["h"])
            iou = calculate_iou(det_box, slot_box)
            if iou > best_iou:
                best_iou, best_slot = iou, sid
        
        if best_slot:
            detected_status = CLASS_NAMES.get(cls, "unknown")
            
            # PERBAIKAN 2: PAKSA override untuk detection "terisi" dengan conf tinggi
            if detected_status == "terisi":
                # Jika detection confidence > 0.4, PAKSA override apapun statusnya
                if conf > 0.4:
                    if slot_status[best_slot] == "reserved":
                        print(f"🚗 FORCE OVERRIDE: {best_slot} reserved -> terisi (conf={conf:.2f}, iou={best_iou:.2f})")
                    slot_status[best_slot] = "terisi"
                    slot_conf[best_slot] = conf
                # Jika conf lebih tinggi dari slot_conf saat ini
                elif conf > slot_conf[best_slot]:
                    slot_status[best_slot] = "terisi"
                    slot_conf[best_slot] = conf
            
            # Untuk detection "kosong", hanya update jika bukan reserved atau conf sangat tinggi
            elif detected_status == "kosong":
                if slot_status[best_slot] != "reserved" or conf > 0.8:
                    if conf > slot_conf[best_slot]:
                        slot_status[best_slot] = detected_status
                        slot_conf[best_slot] = conf
    
    return slot_status

def crop_roi(frame, poly):
    """Crop frame by ROI"""
    x, y, w, h = cv2.boundingRect(poly)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    crop = cv2.bitwise_and(frame, frame, mask=mask)[y:y+h, x:x+w]
    return crop, (x, y)

def draw_slots(frame, slot_zones, slot_status, offset=(0,0)):
    """Draw slot zones - KUNING untuk reserved dengan label 'DIPESAN'"""
    ox, oy = offset
    
    # Warna untuk setiap status
    colors = {
        "kosong": (0, 255, 0),       # Hijau
        "terisi": (0, 0, 255),       # Merah
        "reserved": (0, 255, 255),   # KUNING (Yellow dalam BGR)
        "unknown": (128, 128, 128)   # Abu-abu
    }
    
    # Label untuk ditampilkan
    labels = {
        "kosong": "kosong",
        "terisi": "terisi",
        "reserved": "DIPESAN",
        "unknown": "unknown"
    }
    
    for sid, zone in slot_zones.items():
        x1, y1 = int(zone["x"] + ox), int(zone["y"] + oy)
        x2, y2 = x1 + zone["w"], y1 + zone["h"]
        status = slot_status.get(sid, "unknown")
        color = colors.get(status, (128, 128, 128))
        label = labels.get(status, status)
        
        # Gambar kotak slot
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label slot + status
        cv2.putText(frame, f"{sid}: {label}", (x1+5, y1+18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def api_worker():
    """Background thread untuk kirim API"""
    consecutive_errors = 0
    success_count = 0
    while True:
        try:
            data = api_queue.get(timeout=1)
            
            # Exit signal
            if data is None:
                print(f"\n📊 API Stats: {success_count} successful updates sent")
                api_queue.task_done()
                break
            
            # Hitung statistik
            kosong = sum(1 for v in data.values() if v == "kosong")
            terisi = sum(1 for v in data.values() if v == "terisi")
            reserved = sum(1 for v in data.values() if v == "reserved")
            
            try:
                response = requests.post(LARAVEL_API, json={"slots": data}, 
                                       headers={"Content-Type": "application/json"}, 
                                       timeout=10.0)
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"✅ API UPDATE #{success_count} | {time.strftime('%H:%M:%S')} | Kosong:{kosong} Terisi:{terisi} Dipesan:{reserved}")
                    consecutive_errors = 0
                else:
                    print(f"⚠️  API returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"❌ API TIMEOUT ({consecutive_errors}/3) | {time.strftime('%H:%M:%S')}")
                    
            except requests.exceptions.ConnectionError:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"❌ CANNOT CONNECT TO LARAVEL | Check: php artisan serve | {time.strftime('%H:%M:%S')}")
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 2:
                    print(f"❌ API ERROR: {str(e)[:80]} | {time.strftime('%H:%M:%S')}")
            
            # Always call task_done after processing
            api_queue.task_done()
            
        except:
            # Timeout on queue.get() - no task to mark done
            continue

def send_api_async(slot_status):
    """Queue API request"""
    try:
        if not api_queue.full():
            api_queue.put_nowait(slot_status.copy())
    except:
        pass

# ============== CALIBRATION ==============

def calibrate_slots():
    """Interactive slot calibration tool"""
    
    print("\n" + "="*60)
    print("📍 SLOT CALIBRATION MODE")
    print("="*60)
    
    # Open camera
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ Cannot open camera stream")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        cap.release()
        return False
    
    # Resize to detection resolution
    frame = cv2.resize(frame, (640, 480))
    
    # Crop ROI
    crop, (ox, oy) = crop_roi(frame, ROI_POLY)
    
    # Save screenshot for reference
    cv2.imwrite("calibration_crop.jpg", crop)
    print(f"✅ Screenshot saved: calibration_crop.jpg")
    print(f"   Crop size: {crop.shape[1]}x{crop.shape[0]} (w x h)\n")
    
    print("Instruksi:")
    print("  1. Klik pojok KIRI-ATAS slot")
    print("  2. Klik pojok KANAN-BAWAH slot")
    print("  3. Ulangi untuk semua slot")
    print("  4. Press 's' untuk SAVE")
    print("  5. Press 'r' untuk RESET slot terakhir")
    print("  6. Press 'q' untuk QUIT")
    print("="*60 + "\n")
    
    # State
    slots = {}
    points = []
    current_idx = 0
    display = crop.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, slots, current_idx, display
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            
            # Draw point
            cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow("Calibration", display)
            
            # Complete rectangle
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                
                # Calculate box
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                if current_idx < len(SLOT_NAMES):
                    slot_id = SLOT_NAMES[current_idx]
                    slots[slot_id] = {
                        "x": int(x_min),
                        "y": int(y_min),
                        "w": int(w),
                        "h": int(h)
                    }
                    
                    # Draw rectangle
                    cv2.rectangle(display, (x_min, y_min), (x_min + w, y_min + h),
                                (0, 255, 255), 2)
                    
                    # Label
                    cv2.putText(display, slot_id, (x_min + 5, y_min + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    print(f"✅ {slot_id}: x={x_min}, y={y_min}, w={w}, h={h}")
                    current_idx += 1
                    
                    if current_idx >= len(SLOT_NAMES):
                        print("\n🎉 All slots calibrated!")
                        print("   Press 's' to SAVE or 'q' to QUIT\n")
                
                points = []
                cv2.imshow("Calibration", display)
    
    # Setup window
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", display)
    
    # Main loop
    saved = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Save
        if key == ord('s'):
            if slots:
                with open(SLOT_MAP_JSON, 'w', encoding='utf-8') as f:
                    json.dump(slots, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Slot map saved to: {SLOT_MAP_JSON}")
                print(f"   Total slots: {len(slots)}\n")
                saved = True
            else:
                print("\n⚠️  No slots to save!\n")
        
        # Reset last slot
        elif key == ord('r'):
            if current_idx > 0:
                current_idx -= 1
                last_slot = SLOT_NAMES[current_idx]
                if last_slot in slots:
                    del slots[last_slot]
                
                # Redraw
                display = crop.copy()
                for sid, zone in slots.items():
                    x, y, w, h = zone["x"], zone["y"], zone["w"], zone["h"]
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(display, sid, (x + 5, y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Calibration", display)
                print(f"↩️  Reset {last_slot}")
                points = []
        
        # Quit
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    # Final save if not saved yet
    if not saved and slots:
        with open(SLOT_MAP_JSON, 'w', encoding='utf-8') as f:
            json.dump(slots, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Slot map saved: {SLOT_MAP_JSON}")
        saved = True
    
    return saved

# ============== DETECTION ==============

def run_detection():
    """Main detection loop dengan INSTANT OVERRIDE untuk reserved slots"""
    print("\n🚗 PARKING DETECTION SYSTEM (INSTANT OVERRIDE MODE)")
    print("="*60)
    
    # Load slot map
    if not SLOT_MAP_JSON.exists():
        print(f"❌ Slot map not found: {SLOT_MAP_JSON}")
        return
    
    with open(SLOT_MAP_JSON, 'r') as f:
        slot_zones = json.load(f)
    
    print(f"✅ Loaded {len(slot_zones)} slots")
    
    # Load model
    print("🔥 Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")
    
    # Start API thread
    Thread(target=api_worker, daemon=True).start()
    
    # Open camera
    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera connected")
    print("\n📋 LEGEND:")
    print("   🟢 HIJAU   = Kosong (available)")
    print("   🔴 MERAH   = Terisi (occupied)")
    print("   🟡 KUNING  = DIPESAN (reserved - auto expire 3 menit)")
    print("   ⚪ ABU-ABU = Unknown")
    print("\n🔧 SETTINGS:")
    print(f"   API Update: Every 3 seconds")
    print(f"   Reserve Check: Every 10 seconds")
    print(f"   Detection FPS: ~10 fps (every 3 frames)")
    print(f"   ⚡ INSTANT OVERRIDE: Reserved -> Terisi (NO TIMEOUT)")
    print("\nPress ESC to quit\n")
    print("="*60 + "\n")
    
    # Main loop
    frame_count = 0
    last_api_time = time.time()
    last_reserve_check = time.time()
    last_stats_print = time.time()
    slot_status = {sid: "unknown" for sid in slot_zones.keys()}
    ox, oy = 0, 0
    
    # Fetch reserved slots saat start
    print("🔍 Checking for reserved slots...")
    fetch_reserved_slots()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        
        # Cek reserved slots dari database setiap 10 detik
        if time.time() - last_reserve_check >= 10.0:
            fetch_reserved_slots()
            last_reserve_check = time.time()
        
        # Inference every 3 frames
        if frame_count % 3 == 0:
            crop, (ox, oy) = crop_roi(frame, ROI_POLY)
            results = model.predict(crop, conf=0.3, iou=0.5, imgsz=416, verbose=False)[0]
            slot_status = process_detections(results, slot_zones, reserved_slots_cache)
        
        # Send API every 3 seconds
        if time.time() - last_api_time >= 3.0:
            send_api_async(slot_status)
            last_api_time = time.time()
        
        # Display statistics (setiap 5 detik)
        available = sum(1 for s in slot_status.values() if s == "kosong")
        occupied = sum(1 for s in slot_status.values() if s == "terisi")
        reserved = sum(1 for s in slot_status.values() if s == "reserved")
        
        if time.time() - last_stats_print >= 5.0:
            print(f"📊 Frame {frame_count:5d} | 🟢 Kosong: {available:2d} | 🔴 Terisi: {occupied:2d} | 🟡 Dipesan: {reserved:2d}")
            last_stats_print = time.time()
        
        # Draw visualization
        vis = draw_slots(frame, slot_zones, slot_status, (ox, oy))
        
        # Info header
        info_text = f"Kosong: {available} | Terisi: {occupied} | Dipesan: {reserved}"
        cv2.putText(vis, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Parking Detection - INSTANT OVERRIDE MODE", vis)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    api_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detection stopped!")

# ============== MAIN ==============

def main():
    """Main entry point"""
    
    # Check if slot map exists
    if not SLOT_MAP_JSON.exists():
        print("\n⚠️  Slot map tidak ditemukan!")
        print("Pilih mode:")
        print("  1. Kalibrasi slot (buat slot map)")
        print("  2. Keluar")
        
        choice = input("\nPilihan (1/2): ").strip()
        
        if choice == "1":
            print("\n🔧 Memulai kalibrasi...")
            if calibrate_slots():
                print("\n✅ Kalibrasi selesai!")
                print("Lanjut ke detection? (y/n): ", end="")
                if input().strip().lower() == 'y':
                    run_detection()
            else:
                print("❌ Kalibrasi gagal!")
        else:
            print("👋 Keluar...")
    else:
        print("\nSlot map ditemukan!")
        print("Pilih mode:")
        print("  1. Run detection (INSTANT OVERRIDE MODE)")
        print("  2. Re-kalibrasi slot")
        print("  3. Keluar")
        
        choice = input("\nPilihan (1/2/3): ").strip()
        
        if choice == "1":
            run_detection()
        elif choice == "2":
            print("\n🔧 Memulai re-kalibrasi...")
            if calibrate_slots():
                print("\n✅ Re-kalibrasi selesai!")
                print("Lanjut ke detection? (y/n): ", end="")
                if input().strip().lower() == 'y':
                    run_detection()
        else:
            print("👋 Keluar...")

if __name__ == "__main__":
    main()