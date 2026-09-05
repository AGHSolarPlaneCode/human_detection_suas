import os
import cv2
import random
from pathlib import Path

# --- Configuration ---
INPUT_IMAGES_DIR = r"C:\Users\Bartek\Desktop\SUAV\testy_dataset\images"  # Should contain train/, val/, test/
INPUT_LABELS_DIR = r"C:\Users\Bartek\Desktop\SUAV\testy_dataset\labels"  # Should contain train/, val/, test/ (from previous script)
OUTPUT_DIR = r"C:\Users\Bartek\Desktop\SUAV\testy_dataset\sliced"      # Where the new 640x640 dataset will be saved
TILE_SIZE = 640

def read_yolo_labels(label_path, img_w, img_h):
    """Reads YOLO labels and converts normalized coordinates to absolute pixels."""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # Convert normalized YOLO to absolute pixel coordinates
            abs_w, abs_h = w * img_w, h * img_h
            xmin = (cx * img_w) - (abs_w / 2)
            xmax = (cx * img_w) + (abs_w / 2)
            ymin = (cy * img_h) - (abs_h / 2)
            ymax = (cy * img_h) + (abs_h / 2)
            
            # Clip to image boundaries just in case
            xmin, xmax = max(0, xmin), min(img_w, xmax)
            ymin, ymax = max(0, ymin), min(img_h, ymax)
            
            bboxes.append({'class': class_id, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    return bboxes

def process_split(split):
    in_img_dir = Path(INPUT_IMAGES_DIR) / split
    in_lbl_dir = Path(INPUT_LABELS_DIR) / split
    out_img_dir = Path(OUTPUT_DIR) / 'images' / split
    out_lbl_dir = Path(OUTPUT_DIR) / 'labels' / split
    
    if not in_img_dir.exists():
        return

    # Create output directories
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    total_obj_tiles = 0
    total_bg_tiles = 0

    print(f"Processing split: {split}")
    
    for img_path in in_img_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Pad image if it's smaller than the TILE_SIZE
        pad_h = max(0, TILE_SIZE - img_h)
        pad_w = max(0, TILE_SIZE - img_w)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            img_h, img_w = img.shape[:2] # Update sizes after padding

        label_path = in_lbl_dir / (img_path.stem + '.txt')
        bboxes = read_yolo_labels(label_path, img_w, img_h)
        
        covered_indices = set()
        obj_windows = []
        
        # 1. FIND OBJECT TILES
        for i, box in enumerate(bboxes):
            if i in covered_indices:
                continue
                
            box_w = box['xmax'] - box['xmin']
            box_h = box['ymax'] - box['ymin']
            
            # If the object itself is larger than the tile, we can't fit it without cutting. Skip it.
            if box_w > TILE_SIZE or box_h > TILE_SIZE:
                continue
                
            # Define limits for top-left (x0, y0) of our 640x640 window so it completely encloses the box
            min_x0 = max(0, int(box['xmax']) - TILE_SIZE)
            max_x0 = min(img_w - TILE_SIZE, int(box['xmin']))
            min_y0 = max(0, int(box['ymax']) - TILE_SIZE)
            max_y0 = min(img_h - TILE_SIZE, int(box['ymin']))
            
            # Try 30 random positions to find a "clean" window (doesn't cut OTHER boxes)
            for _ in range(30):
                x0 = random.randint(min_x0, max_x0)
                y0 = random.randint(min_y0, max_y0)
                x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
                
                cut_any = False
                contained = []
                
                for j, b in enumerate(bboxes):
                    overlap = (b['xmin'] < x1 and b['xmax'] > x0 and b['ymin'] < y1 and b['ymax'] > y0)
                    inside = (b['xmin'] >= x0 and b['xmax'] <= x1 and b['ymin'] >= y0 and b['ymax'] <= y1)
                    
                    if overlap and not inside:
                        cut_any = True # A box is partially inside, meaning it got cut!
                        break
                    if inside:
                        contained.append((j, b))
                
                if not cut_any:
                    # Valid tile found!
                    obj_windows.append((x0, y0, contained))
                    for j, _ in contained:
                        covered_indices.add(j)
                    break
                    
        # 2. FIND BACKGROUND TILES (Aiming for 1:1 ratio with object tiles)
        bg_windows = []
        # If the image has no objects, still extract 2 bg tiles so the model learns pure background.
        target_bg_count = len(obj_windows) if len(obj_windows) > 0 else 2 
        
        attempts = 0
        while len(bg_windows) < target_bg_count and attempts < 100:
            attempts += 1
            x0 = random.randint(0, img_w - TILE_SIZE)
            y0 = random.randint(0, img_h - TILE_SIZE)
            x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
            
            overlap_any = False
            for b in bboxes:
                # If window overlaps with ANY bounding box, it's not a pure background
                if (b['xmin'] < x1 and b['xmax'] > x0 and b['ymin'] < y1 and b['ymax'] > y0):
                    overlap_any = True
                    break
            
            if not overlap_any:
                bg_windows.append((x0, y0, []))

        # 3. SAVE CROP IMAGES AND RECALCULATED LABELS
        all_windows = obj_windows + bg_windows
        
        for idx, (x0, y0, contained_boxes) in enumerate(all_windows):
            is_bg = len(contained_boxes) == 0
            tile_type = "bg" if is_bg else "obj"
            
            # Crop image
            tile = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
            
            # Save names format: originalname_obj_x0_y0.jpg
            base_name = f"{img_path.stem}_{tile_type}_{x0}_{y0}"
            tile_img_path = out_img_dir / (base_name + img_path.suffix)
            tile_lbl_path = out_lbl_dir / (base_name + '.txt')
            
            cv2.imwrite(str(tile_img_path), tile)
            
            if is_bg:
                # Save empty text file for background tile (required by YOLO)
                open(tile_lbl_path, 'w').close()
                total_bg_tiles += 1
            else:
                # Recalculate YOLO bounds based on the 640x640 tile
                with open(tile_lbl_path, 'w') as f:
                    for _, b in contained_boxes:
                        # Shift to tile coordinates
                        new_xmin = b['xmin'] - x0
                        new_xmax = b['xmax'] - x0
                        new_ymin = b['ymin'] - y0
                        new_ymax = b['ymax'] - y0
                        
                        # Normalize YOLO coordinates (0.0 to 1.0)
                        new_cx = ((new_xmin + new_xmax) / 2) / TILE_SIZE
                        new_cy = ((new_ymin + new_ymax) / 2) / TILE_SIZE
                        new_w = (new_xmax - new_xmin) / TILE_SIZE
                        new_h = (new_ymax - new_ymin) / TILE_SIZE
                        
                        f.write(f"{b['class']} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
                total_obj_tiles += 1

    print(f"  -> Generated {total_obj_tiles} object tiles and {total_bg_tiles} background tiles.\n")

if __name__ == "__main__":
    for split_name in ['train', 'val', 'test']:
        process_split(split_name)
    print("Dataset slicing complete! Saved to:", OUTPUT_DIR)