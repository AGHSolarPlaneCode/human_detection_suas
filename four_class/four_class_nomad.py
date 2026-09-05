

import os
import cv2
import json
import random
from pathlib import Path

# --- Configuration ---
INPUT_IMAGES_DIR = r"C:\Users\Bartek\Desktop\SUAV\whole_nomad"  # Folder containing the Actor folders
JSON_ANNOTATIONS_PATH = r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\annotations (2).json"
JSON_ACTIVITY_PATH = r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\activityLabels (1).json"
OUTPUT_DIR = r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\four_class_dataset"  # Output directory for the processed dataset  
TILE_SIZE = 640

def load_activity_labels(json_path):
    """Loads activity labels and safely handles typos in the manual JSON annotations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    lookup = {}
    for item in data:
        actor_id = item['id']
        lookup[actor_id] = {}
        for alt_str, actions in item['labels'].items():
            alt_int = int(alt_str)
            lookup[actor_id][alt_int] = {}
            for action, ranges in actions.items():
                for r in ranges:
                    if not r:
                        continue
                        
                    # Fix Typo 1: ["0174.0334"] instead of ["0174", "0334"]
                    if len(r) == 1 and '.' in str(r[0]):
                        r = str(r[0]).split('.')
                    # Fix Typo 2: missing end frame e.g. ["0728"]
                    elif len(r) == 1:
                        r = [r[0], r[0]]
                        
                    try:
                        # Fix Typo 3: accidental characters like "1320}"
                        start_str = ''.join(filter(str.isdigit, str(r[0])))
                        end_str = ''.join(filter(str.isdigit, str(r[1])))
                        
                        start_f = int(start_str)
                        end_f = int(end_str)
                    except ValueError:
                        # Commented out warning to keep progress bar clean
                        # print(f"\nWarning: Skipping malformed range {r} for Actor {actor_id}")
                        continue
                        
                    # Fix Typo 4: Ensure range always goes from lowest to highest
                    if start_f > end_f:
                        start_f, end_f = end_f, start_f
                        
                    for f in range(start_f, end_f + 1):
                        lookup[actor_id][alt_int][f] = action
    return lookup

def load_annotations(json_path):
    """Loads bounding boxes and visibility."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    anno_dict = {}
    for item in data:
        anno_dict[item['file_name']] = item.get('annotations', [])
    return anno_dict

def get_class_id(visibility, filename, activity_lookup):
    """
    Determines the class based on visibility and activity.
    Returns:
       0: covered (50 < visibility <= 80)
       1: lying (visibility > 80 AND action is laying)
       2: standing (visibility > 80 AND other actions)
    """
    if 50 < visibility <= 80:
        return 0
    
    # For visibility 90 - 100, parse filename to look up activity
    stem = Path(filename).stem
    parts = stem.split('_')
    
    try:
        actor_id = int(parts[0].replace('Actor', ''))
        alt = int(parts[1].replace('a', ''))
        frame = int(parts[2].replace('f', ''))
        
        # Lookup the action for this specific frame
        action = activity_lookup.get(actor_id, {}).get(alt, {}).get(frame, None)
        
        if action in ["Laying", "Hiding (Laying)"]:
            return 1
            
    except (IndexError, ValueError):
        pass
        
    return 2 # Default to standing for all other >80 visibility cases

def get_split(actor_num):
    """Returns the split based on actor number."""
    if 1 <= actor_num <= 80:
        return 'train'
    elif 81 <= actor_num <= 90:
        return 'val'
    elif 91 <= actor_num <= 100:
        return 'test'
    return None

def process_dataset():
    out_dir = Path(OUTPUT_DIR)
    
    # Load JSON files into memory
    print("Loading annotations and activity labels...")
    anno_dict = load_annotations(JSON_ANNOTATIONS_PATH)
    activity_lookup = load_activity_labels(JSON_ACTIVITY_PATH)
    
    stats = {'train': {'obj': 0, 'bg': 0}, 'val': {'obj': 0, 'bg': 0}, 'test': {'obj': 0, 'bg': 0}}
    
    image_paths = list(Path(INPUT_IMAGES_DIR).rglob('*.[jJ][pP][gG]'))
    total_images = len(image_paths)
    print(f"Found {total_images} images total. Applying filters and slicing...\n")
    
    # --- ADDED ENUMERATE FOR PROGRESS METRIC ---
    for idx, img_path in enumerate(image_paths):
        # Progress Metric (overwrites the current console line)
        progress_pct = (idx + 1) / total_images * 100
        print(f"\rProcessing image {idx + 1}/{total_images} ({progress_pct:.1f}%)", end="", flush=True)

        filename = img_path.name
        
        # --- HEIGHT FILTER ---
        if '_a70_' not in filename and '_a90_' not in filename:
            continue
        
        # Determine the split based on the Actor number in the filename
        try:
            actor_str = filename.split('_')[0]
            actor_num = int(actor_str.replace('Actor', ''))
        except (IndexError, ValueError):
            continue
            
        split = get_split(actor_num)
        if not split:
            continue
            
        # Create output directories for this split
        out_img_dir = out_dir / 'images' / split
        out_lbl_dir = out_dir / 'labels' / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Pad if image is smaller than 640x640
        pad_h = max(0, TILE_SIZE - img_h)
        pad_w = max(0, TILE_SIZE - img_w)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            img_h, img_w = img.shape[:2]
        
        # Extract bboxes and set class IDs
        bboxes = []
        for ann in anno_dict.get(filename, []):
            visibility = int(ann.get('visibility', 100))
            
            # --- VISIBILITY FILTER ---
            # Completely ignore objects with visibility 50 or less
            if visibility <= 50:
                continue
                
            x_min, y_min, w, h = ann['bbox']
            cls_id = get_class_id(visibility, filename, activity_lookup)
            
            bboxes.append({
                'class': cls_id, 
                'xmin': x_min, 
                'ymin': y_min, 
                'xmax': x_min + w, 
                'ymax': y_min + h
            })

        covered_indices = set()
        obj_windows = []
        
        # 1. FIND OBJECT TILES
        for i, box in enumerate(bboxes):
            if i in covered_indices:
                continue
                
            box_w = box['xmax'] - box['xmin']
            box_h = box['ymax'] - box['ymin']
            
            if box_w > TILE_SIZE or box_h > TILE_SIZE:
                continue
                
            min_x0 = max(0, int(box['xmax']) - TILE_SIZE)
            max_x0 = min(img_w - TILE_SIZE, int(box['xmin']))
            min_y0 = max(0, int(box['ymax']) - TILE_SIZE)
            max_y0 = min(img_h - TILE_SIZE, int(box['ymin']))
            
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
                        cut_any = True
                        break
                    if inside:
                        contained.append((j, b))
                
                if not cut_any:
                    obj_windows.append((x0, y0, contained))
                    for j, _ in contained:
                        covered_indices.add(j)
                    break
                    
        # 2. FIND BACKGROUND TILES
        bg_windows = []
        target_bg_count = len(obj_windows) if len(obj_windows) > 0 else 2 
        
        attempts = 0
        while len(bg_windows) < target_bg_count and attempts < 100:
            attempts += 1
            x0 = random.randint(0, img_w - TILE_SIZE)
            y0 = random.randint(0, img_h - TILE_SIZE)
            x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
            
            overlap_any = False
            for b in bboxes:
                if (b['xmin'] < x1 and b['xmax'] > x0 and b['ymin'] < y1 and b['ymax'] > y0):
                    overlap_any = True
                    break
            
            if not overlap_any:
                bg_windows.append((x0, y0, []))

        # 3. CROP AND SAVE
        all_windows = obj_windows + bg_windows
        
        for idx_window, (x0, y0, contained_boxes) in enumerate(all_windows):
            is_bg = len(contained_boxes) == 0
            tile_type = "bg" if is_bg else "obj"
            
            tile = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
            base_name = f"{img_path.stem}_{tile_type}_{x0}_{y0}"
            tile_img_path = out_img_dir / (base_name + img_path.suffix)
            tile_lbl_path = out_lbl_dir / (base_name + '.txt')
            
            cv2.imwrite(str(tile_img_path), tile)
            
            if is_bg:
                open(tile_lbl_path, 'w').close()
                stats[split]['bg'] += 1
            else:
                with open(tile_lbl_path, 'w') as f:
                    for _, b in contained_boxes:
                        new_xmin = b['xmin'] - x0
                        new_xmax = b['xmax'] - x0
                        new_ymin = b['ymin'] - y0
                        new_ymax = b['ymax'] - y0
                        
                        new_cx = ((new_xmin + new_xmax) / 2) / TILE_SIZE
                        new_cy = ((new_ymin + new_ymax) / 2) / TILE_SIZE
                        new_w = (new_xmax - new_xmin) / TILE_SIZE
                        new_h = (new_ymax - new_ymin) / TILE_SIZE
                        
                        f.write(f"{b['class']} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
                stats[split]['obj'] += 1

    # Added newlines here so the summary doesn't overwrite the progress bar
    print("\n\n--- Summary ---")
    for s in ['train', 'val', 'test']:
        print(f"{s.upper()}: Generated {stats[s]['obj']} object tiles and {stats[s]['bg']} background tiles.")
    print("\nComplete! Dataset saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    process_dataset()