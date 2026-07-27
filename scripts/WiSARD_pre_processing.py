import os
import random
import shutil
import cv2
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path(r"C:\Users\Bartek\Desktop\SUAV\wisard")  # Replace with your unzipped dataset path
OUTPUT_DIR = Path(r"C:\Users\Bartek\Desktop\SUAV\wisard_preprocessed")  # Output directory for YOLOv5/v8 dataset
CROP_SIZE = 640
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # Train, Val, Test

def create_dirs():
    """Creates the YOLOv5/v8 expected directory structure."""
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def read_yolo_labels(label_path, img_w, img_h):
    """Reads YOLO labels and converts normalized coordinates to absolute pixels."""
    bboxes = []
    if not label_path.exists():
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                
                x_min = int((xc - w / 2) * img_w)
                y_min = int((yc - h / 2) * img_h)
                x_max = int((xc + w / 2) * img_w)
                y_max = int((yc + h / 2) * img_h)
                
                bboxes.append([cls_id, x_min, y_min, x_max, y_max])
    return bboxes

def generate_crops(image, bboxes, img_name):
    """
    Generates 640x640 crops ensuring target actors are fully visible and 
    somewhat randomly positioned without being cut in half.
    """
    img_h, img_w = image.shape[:2]
    covered = [False] * len(bboxes)
    crops_data = []

    for i, target_box in enumerate(bboxes):
        if covered[i]:
            continue
            
        cls_id, tx_min, ty_min, tx_max, ty_max = target_box
        
        box_w = tx_max - tx_min
        box_h = ty_max - ty_min
        if box_w > CROP_SIZE or box_h > CROP_SIZE:
            print(f"Skipping a box in {img_name} - larger than {CROP_SIZE}x{CROP_SIZE}")
            covered[i] = True
            continue

        min_cx = max(0, tx_max - CROP_SIZE)
        max_cx = min(img_w - CROP_SIZE, tx_min)
        min_cy = max(0, ty_max - CROP_SIZE)
        max_cy = min(img_h - CROP_SIZE, ty_min)
        
        crop_x = random.randint(min_cx, max_cx) if max_cx >= min_cx else 0
        crop_y = random.randint(min_cy, max_cy) if max_cy >= min_cy else 0
        
        crop_img = image[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]
        
        crop_labels = []
        for j, box in enumerate(bboxes):
            b_cls, bx_min, by_min, bx_max, by_max = box
            
            if (bx_min >= crop_x and bx_max <= crop_x + CROP_SIZE and 
                by_min >= crop_y and by_max <= crop_y + CROP_SIZE):
                
                local_xmin = bx_min - crop_x
                local_ymin = by_min - crop_y
                local_xmax = bx_max - crop_x
                local_ymax = by_max - crop_y
                
                local_xc = ((local_xmin + local_xmax) / 2.0) / CROP_SIZE
                local_yc = ((local_ymin + local_ymax) / 2.0) / CROP_SIZE
                local_w = (local_xmax - local_xmin) / CROP_SIZE
                local_h = (local_ymax - local_ymin) / CROP_SIZE
                
                crop_labels.append(f"{b_cls} {local_xc:.6f} {local_yc:.6f} {local_w:.6f} {local_h:.6f}")
                covered[j] = True 
                
        crops_data.append((crop_img, crop_labels))
        
    return crops_data

def main():
    print("Starting preprocessing...")
    create_dirs()
    
    all_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir() and "VIS" in f.name]
    
    if not all_folders:
        print("No 'VIS' folders found. Please check your SOURCE_DIR.")
        return

    random.seed(42)
    random.shuffle(all_folders)
    
    total_folders = len(all_folders)
    train_idx = int(total_folders * SPLIT_RATIOS[0])
    val_idx = train_idx + int(total_folders * SPLIT_RATIOS[1])
    
    splits = {
        'train': all_folders[:train_idx],
        'val': all_folders[train_idx:val_idx],
        'test': all_folders[val_idx:]
    }

    for split_name, folders in splits.items():
        print(f"--- Processing {split_name} split ({len(folders)} folders) ---")
        
        for folder in folders:
            image_paths = list(folder.rglob('*.jpg')) + list(folder.rglob('*.jpeg'))
            
            for img_path in image_paths:
                label_path = img_path.with_suffix('.txt')
                
                if not label_path.exists():
                    continue
                
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                img_h, img_w = image.shape[:2]
                
                bboxes = read_yolo_labels(label_path, img_w, img_h)
                
                if not bboxes:
                    continue
                
                crops = generate_crops(image, bboxes, img_path.name)
                
                for idx, (crop_img, crop_labels) in enumerate(crops):
                    base_name = f"{img_path.stem}_crop{idx}"
                    out_img_path = OUTPUT_DIR / 'images' / split_name / f"{base_name}.jpg"
                    out_lbl_path = OUTPUT_DIR / 'labels' / split_name / f"{base_name}.txt"

                    cv2.imwrite(str(out_img_path), crop_img)

                    with open(out_lbl_path, 'w') as f:
                        f.write('\n'.join(crop_labels))

    print(f"Preprocessing complete! Dataset saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()