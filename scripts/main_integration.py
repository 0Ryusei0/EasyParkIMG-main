import sys
import os
import time
import cv2
import json
import requests
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============== PATH SETUP ==============
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ============== CONFIG ==============
STREAM_URL = "http://192.168.1.11:4747/video"
MODEL_PATH = r"D:\EasyParkIMG-main\Model\parking_detection2\weights\best.pt"
SLOT_MAP_JSON = ROOT / "slot_map.json"
LARAVEL_API = "http://localhost:8000/api/parking/update"  # Ganti dengan URL Laravel Anda

# ROI Polygon (sesuai code sebelumnya)
ROI_POLY = np.array([
    [120, 90],
    [520, 80],
    [560, 460],
    [100, 470],
], dtype=np.int32)

# Class names dari model YOLO
CLASS_NAMES = {0: "terisi", 1: "kosong"}

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
    """
    Process YOLO detections dan build slot status
    
    Returns:
        dict: {
            "slots": {"A1": "kosong", "A2": "terisi", ...},
            "details": {...},
            "summary": {...}
        }
    """
    # Initialize
    slot_status = {slot_id: "unknown" for slot_id in slot_zones.keys()}
    slot_details = {}
    slot_best_conf = {slot_id: 0.0 for slot_id in slot_zones.keys()}

    # Check if results have boxes
    if not hasattr(yolo_results, 'boxes') or len(yolo_results.boxes) == 0:
        summary = {
            "total": len(slot_zones),
            "available": 0,
            "occupied": 0,
            "unknown": len(slot_zones)
        }
        return {"slots": slot_status, "details": slot_details, "summary": summary}

    # Process each detection
    for box in yolo_results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Filter: confidence
        if conf < conf_thresh:
            continue

        # Filter: minimum area
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        # Map to slot
        det_box = (int(x1), int(y1), int(x2), int(y2))
        slot_id, iou = map_detection_to_slot(det_box, slot_zones, min_iou=min_iou)

        if slot_id is None:
            continue

        # Keep highest confidence per slot
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

    # Summary
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
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{slot_id}: {status}"
        cv2.putText(frame, label, (x1 + 5, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


def send_to_laravel(slot_status):
    """
    Kirim status slot ke Laravel API
    
    Args:
        slot_status: dict {"A1": "kosong", "A2": "terisi", ...}
    
    Returns:
        tuple: (status_code, response_text)
    """
    try:
        payload = {"slots": slot_status}
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer YOUR_TOKEN"  # Uncomment jika pakai auth
        }
        
        response = requests.post(
            LARAVEL_API,
            json=payload,
            headers=headers,
            timeout=2.0
        )
        
        return response.status_code, response.text
    except Exception as e:
        return None, str(e)


def load_slot_map(path):
    """Load slot map dari JSON"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Slot map not found: {path}")
    
    with open(path, 'r') as f:
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
    print("  2. Ulangi untuk semua 10 slot (A1-A5, B1-B5)")
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
                    
                    # Draw rectangle
                    cv2.rectangle(crop, (x_slot, y_slot),
                                (x_slot + w_slot, y_slot + h_slot),
                                (0, 255, 255), 2)
                    
                    # Label
                    cv2.putText(crop, slot_id, (x_slot + 5, y_slot + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    print(f"✅ {slot_id}: x={x_slot}, y={y_slot}, w={w_slot}, h={h_slot}")
                    current_idx += 1
                    
                    if current_idx >= len(slot_names):
                        print("\n🎉 Semua 10 slot sudah dikalibrasi!")
                        print("   Press 's' untuk save atau 'q' untuk quit")
                
                points = []
                cv2.imshow("Calibration", crop)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_cb)
    cv2.imshow("Calibration", crop)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            with open(save_path, 'w') as f:
                json.dump(slots, f, indent=2)
            print(f"\n✅ Slot map saved to: {save_path}")
        
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    # Final save
    if slots:
        with open(save_path, 'w') as f:
            json.dump(slots, f, indent=2)
        print(f"\n✅ Final slot map saved to: {save_path}")


# ============== MAIN ==============

def main():
    print("\n" + "="*60)
    print("🚗 PARKING SLOT MAPPING SYSTEM")
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
    slot_zones = load_slot_map(SLOT_MAP_JSON)
    print(f"✅ Loaded {len(slot_zones)} slots: {list(slot_zones.keys())}\n")
    
    # Load model
    print("🔥 Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded\n")
    
    # Camera
    print(f"📸 Connecting to: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)
    time.sleep(1.0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera stream")
        return
    
    print("✅ Camera connected\n")
    print("="*60)
    print("🎬 Starting detection loop...")
    print("   Press ESC to quit")
    print("="*60 + "\n")
    
    # Main loop
    frame_count = 0
    send_interval = 2.0  # Kirim ke Laravel setiap 2 detik
    last_send_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Resize to expected resolution
        frame = cv2.resize(frame, (640, 480))
        
        # Crop ROI
        crop, (ox, oy) = crop_by_roi(frame, ROI_POLY)
        
        # YOLO inference
        results = model.predict(
            crop,
            conf=0.30,
            iou=0.5,
            imgsz=416,
            verbose=False
        )[0]
        
        # Process detections
        status_data = process_detections(
            results,
            slot_zones,
            CLASS_NAMES,
            conf_thresh=0.5,
            min_area=400,
            min_iou=0.25
        )
        
        slot_status = status_data["slots"]
        summary = status_data["summary"]
        
        # Print summary setiap 30 frame
        if frame_count % 30 == 0:
            print(f"\n📊 Frame {frame_count}")
            print(f"   ✅ Available: {summary['available']}")
            print(f"   ❌ Occupied: {summary['occupied']}")
            print(f"   ❓ Unknown: {summary['unknown']}")
            print(f"   Status: {slot_status}")
        
        # Send to Laravel setiap interval (DISABLED - untuk testing)
        # current_time = time.time()
        # if current_time - last_send_time >= send_interval:
        #     status_code, response = send_to_laravel(slot_status)
        #     if status_code == 200:
        #         print(f"✅ Sent to Laravel: {status_code}")
        #     else:
        #         print(f"⚠  Laravel response: {status_code} - {response}")
        #     last_send_time = current_time
        
        # Visualization
        vis = draw_slot_zones(frame, slot_zones, slot_status, roi_offset=(ox, oy))
        
        # Info overlay
        cv2.putText(vis, f"Available: {summary['available']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Occupied: {summary['occupied']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Parking Mapping", vis)
        
        # Quit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    print("\n🛑 Stopping...")
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!\n")


if __name__ == "__main__":
    main()