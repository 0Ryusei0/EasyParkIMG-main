import cv2
import json
import requests
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from threading import Thread
from queue import Queue
import time

# ============== CONFIG ==============
CAMERA_INDEX = 0  # Webcam default (0, 1, 2, dst)
MODEL_PATH = r"D:\EasyParkIMG-main\Model\parking_detection2\weights\best.pt"
SLOT_MAP_JSON = Path(__file__).parent / "slot_map.json"
LARAVEL_API = "http://localhost:8000/api/parking-slots/update"

# ROI Polygon
ROI_POLY = np.array([[120, 90], [520, 80], [560, 460], [100, 470]], dtype=np.int32)

# Class names
CLASS_NAMES = {0: "terisi", 1: "kosong"}

# Slot names
SLOT_NAMES = ["A1", "A2", "A3", "A4", "A5", "A6", 
              "B1", "B2", "B3", "B4", "B5", "B6"]

# Queue untuk async API
api_queue = Queue(maxsize=2)

# ============== FUNCTIONS ==============

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

def process_detections(results, slot_zones, min_iou=0.25):
    """Process YOLO results and map to slots"""
    slot_status = {sid: "unknown" for sid in slot_zones.keys()}
    slot_conf = {sid: 0.0 for sid in slot_zones.keys()}
    
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return slot_status
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        if conf < 0.5 or (x2-x1)*(y2-y1) < 400:
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
        
        if best_slot and conf > slot_conf[best_slot]:
            slot_status[best_slot] = CLASS_NAMES.get(cls, "unknown")
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
    """Draw slot zones with status colors"""
    ox, oy = offset
    colors = {"kosong": (0, 255, 0), "terisi": (0, 0, 255), "unknown": (128, 128, 128)}
    
    for sid, zone in slot_zones.items():
        x1, y1 = int(zone["x"] + ox), int(zone["y"] + oy)
        x2, y2 = x1 + zone["w"], y1 + zone["h"]
        status = slot_status.get(sid, "unknown")
        color = colors.get(status, (128, 128, 128))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{sid}: {status}", (x1+5, y1+18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def api_worker():
    """Background thread untuk kirim API"""
    while True:
        try:
            data = api_queue.get(timeout=1)
            if data is None:
                break
            
            response = requests.post(LARAVEL_API, json={"slots": data}, 
                                   headers={"Content-Type": "application/json"}, 
                                   timeout=3.0)
            
            if response.status_code == 200:
                print(f"✅ API sent at {time.strftime('%H:%M:%S')}")
            
            api_queue.task_done()
        except:
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
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Cannot open webcam (index: {CAMERA_INDEX})")
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
    """Main detection loop"""
    print("\n🚗 PARKING DETECTION SYSTEM (WEBCAM)")
    
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
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"❌ Cannot open webcam (index: {CAMERA_INDEX})")
        return
    
    print(f"✅ Webcam connected (index: {CAMERA_INDEX})\nPress ESC to quit\n")
    
    # Main loop
    frame_count = 0
    last_api_time = time.time()
    slot_status = {sid: "unknown" for sid in slot_zones.keys()}
    ox, oy = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        
        # Inference every 3 frames
        if frame_count % 3 == 0:
            crop, (ox, oy) = crop_roi(frame, ROI_POLY)
            results = model.predict(crop, conf=0.3, iou=0.5, imgsz=416, verbose=False)[0]
            slot_status = process_detections(results, slot_zones)
        
        # Send API every 2 seconds
        if time.time() - last_api_time >= 2.0:
            send_api_async(slot_status)
            last_api_time = time.time()
        
        # Display
        available = sum(1 for s in slot_status.values() if s == "kosong")
        occupied = sum(1 for s in slot_status.values() if s == "terisi")
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | Available: {available} | Occupied: {occupied}")
        
        vis = draw_slots(frame, slot_zones, slot_status, (ox, oy))
        cv2.putText(vis, f"Available: {available} | Occupied: {occupied}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Parking Detection", vis)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    api_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")

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
        print("  1. Run detection")
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