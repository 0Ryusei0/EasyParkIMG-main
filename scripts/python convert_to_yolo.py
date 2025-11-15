import json
import os
from pathlib import Path

def convert_labelme_to_yolo(json_file, output_dir, class_mapping):
    """
    Convert LabelMe JSON to YOLO format
    
    Args:
        json_file: path to .json file from LabelMe
        output_dir: directory to save YOLO .txt files
        class_mapping: dict mapping class names to class IDs
                      e.g., {'terisi': 0, 'kosong': 1}
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_height = data['imageHeight']
        img_width = data['imageWidth']
        
        # Prepare output file
        output_file = os.path.join(output_dir, 
                                   Path(json_file).stem + '.txt')
        
        with open(output_file, 'w') as f:
            for shape in data['shapes']:
                label = shape['label']
                if label not in class_mapping:
                    print(f"Warning: Unknown label '{label}' in {json_file.name}")
                    continue
                
                class_id = class_mapping[label]
                points = shape['points']
                
                # LabelMe rectangle: [[x1,y1], [x2,y2]]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return True
    except Exception as e:
        print(f"Error converting {json_file}: {e}")
        return False

def batch_convert(json_dir, output_dir, class_mapping):
    """Convert all JSON files in a directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = list(Path(json_dir).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Converting to YOLO format...\n")
    
    success_count = 0
    for json_file in json_files:
        if convert_labelme_to_yolo(json_file, output_dir, class_mapping):
            success_count += 1
            print(f"✓ Converted: {json_file.name}")
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {success_count}/{len(json_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

def create_classes_file(output_dir, class_mapping):
    """Create classes.txt file for YOLO"""
    classes_file = os.path.join(output_dir, 'classes.txt')
    
    # Sort by class ID
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    
    with open(classes_file, 'w') as f:
        for class_name, _ in sorted_classes:
            f.write(f"{class_name}\n")
    
    print(f"\n✓ Created classes.txt with {len(class_mapping)} classes")

# ========== KONFIGURASI - SESUAIKAN PATH DI SINI ==========

# Path ke folder yang berisi file .json hasil labeling
JSON_DIRECTORY = r"D:\Quant_ML_Project\ML.py\EasyPark\Dataset\parking_lot_final\valid\labels"

# Path ke folder output untuk menyimpan file .txt YOLO
OUTPUT_DIRECTORY = r"D:\Quant_ML_Project\ML.py\EasyPark\Dataset\parking_lot_final\valid\labels"

# Mapping nama class ke class ID (sesuai dengan label yang kamu pakai)
CLASS_MAPPING = {
    'terisi': 0,
    'kosong': 1
}

# ===========================================================

if __name__ == "__main__":
    print("LabelMe to YOLO Converter")
    print("="*50)
    
    # Convert all JSON files
    batch_convert(JSON_DIRECTORY, OUTPUT_DIRECTORY, CLASS_MAPPING)
    
    # Create classes.txt
    create_classes_file(OUTPUT_DIRECTORY, CLASS_MAPPING)
    
    print("\nDone! You can now use these labels for YOLO training.")