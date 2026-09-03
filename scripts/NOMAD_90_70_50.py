import os
import json
import random
import shutil
import cv2
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(r"C:\Users\Bartek\Desktop\SUAV\whole_nomad")
ACTIVITY_JSON = Path(r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\activityLabels (1).json")
ANNOTATIONS_JSON = Path(r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\annotations (2).json")
OUTPUT_DIR = Path(r"C:\Users\Bartek\Desktop\SUAV\yolo_tiles_640")

CROP_SIZE = 640
VAL_SPLIT_RATIO = 0.2
BG_RATIO = 0.15  # 15% of the FINAL dataset will be background
RANDOM_SEED = 42

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_source_path(file_name, base_dir):
    parts = file_name.replace('.jpg', '').split('_')
    actor_str = parts[0]
    actor_id = int(actor_str.replace('Actor', ''))
    alt_str = parts[1]
    
    group_start = ((actor_id - 1) // 10) * 10 + 1
    group_end = group_start + 9
    group_folder = f"Actor{group_start:03d}-Actor{group_end:03d}"
    
    return base_dir / group_folder / actor_str / f"{actor_str}_{alt_str}" / file_name

def is_fully_inside(bbox, crop_x, crop_y, crop_size):
    bx, by, bw, bh = bbox
    return (bx >= crop_x and by >= crop_y and 
            bx + bw <= crop_x + crop_size and 
            by + bh <= crop_y + crop_size)

def intersects(bbox, crop_x, crop_y, crop_size):
    bx, by, bw, bh = bbox
    # Check if the bounding box and the crop rectangle overlap
    return not (bx + bw <= crop_x or bx >= crop_x + crop_size or 
                by + bh <= crop_y or by >= crop_y + crop_size)

def get_safe_random_crop(target_bbox, all_bboxes, img_w, img_h, crop_size=640, max_retries=50):
    tx, ty, tw, th = target_bbox
    
    # If the actor is larger than the crop size, we can't fit them
    if tw > crop_size or th > crop_size:
        return None, []

    min_cx = max(0, int(tx + tw - crop_size))
    max_cx = min(img_w - crop_size, int(tx))
    min_cy = max(0, int(ty + th - crop_size))
    max_cy = min(img_h - crop_size, int(ty))

    if min_cx > max_cx or min_cy > max_cy:
        return None, []

    for _ in range(max_retries):
        cx = random.randint(min_cx, max_cx)
        cy = random.randint(min_cy, max_cy)
        
        valid = True
        boxes_in_crop = []
        
        for bbox in all_bboxes:
            if is_fully_inside(bbox, cx, cy, crop_size):
                boxes_in_crop.append(bbox)
            elif intersects(bbox, cx, cy, crop_size):
                # Box is partially cut by the crop edge. Invalid crop!
                valid = False
                break
                
        if valid:
            return (cx, cy), boxes_in_crop
            
    return None, []

def get_background_crop(all_bboxes, img_w, img_h, crop_size=640, max_retries=50):
    for _ in range(max_retries):
        cx = random.randint(0, img_w - crop_size)
        cy = random.randint(0, img_h - crop_size)
        
        conflict = False
        for bbox in all_bboxes:
            if intersects(bbox, cx, cy, crop_size):
                conflict = True
                break
                
        if not conflict:
            return (cx, cy)
    return None

def convert_to_yolo(bbox, crop_x, crop_y, crop_size):
    bx, by, bw, bh = bbox
    # Adjust coordinates relative to the crop
    new_x = bx - crop_x
    new_y = by - crop_y
    
    x_center = (new_x + bw / 2.0) / crop_size
    y_center = (new_y + bh / 2.0) / crop_size
    w_norm = bw / crop_size
    h_norm = bh / crop_size
    
    return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    random.seed(RANDOM_SEED)
    
    # 1. ACTOR-BASED TRAIN/VAL SPLIT
    all_actors = list(range(1, 101))
    random.shuffle(all_actors)
    split_idx = int(len(all_actors) * (1 - VAL_SPLIT_RATIO))
    train_actors = set(all_actors[:split_idx])
    val_actors = set(all_actors[split_idx:])
    
    print(f"Train Actors: {len(train_actors)}")
    print(f"Val Actors: {len(val_actors)}")

    # 2. PARSE ACTIVITIES
    pose_map = {}
    with open(ACTIVITY_JSON, 'r') as f:
        activities = json.load(f)
        
    for act in activities:
        act_id = act['id']
        pose_map[act_id] = {}
        for alt, poses in act['labels'].items():
            pose_map[act_id][alt] = {}
            for pose_name, intervals in poses.items():
                for interval in intervals:
                    # 1. Skip completely empty or invalid types
                    if not isinstance(interval, list) or len(interval) == 0:
                        continue
                        
                    # 2. Fix known dataset typos (e.g., ["0174.0334"] -> ["0174", "0334"])
                    if len(interval) == 1 and '.' in str(interval[0]):
                        interval = str(interval[0]).split('.')
                        
                    # 3. Strip accidental characters (e.g., "1320}")
                    interval = [str(x).replace('}', '').replace('{', '').strip() for x in interval]

                    # 4. Assign start and end
                    if len(interval) >= 2:
                        start, end = interval[0], interval[1]
                    elif len(interval) == 1:
                        start = end = interval[0] # Single frame
                        
                    # 5. Safely convert to int, bypassing any remaining fatal typos
                    try:
                        start_idx = int(start)
                        end_idx = int(end)
                    except ValueError:
                        print(f"Skipping malformed annotation in Actor {act_id} ({alt}m): {interval}")
                        continue
                        
                    # Map the poses
                    for f_num in range(start_idx, end_idx + 1):
                        pose_map[act_id][alt][f_num] = pose_name

    # 3. FILTER FRAMES
    print("Parsing annotations and logic filtering...")
    with open(ANNOTATIONS_JSON, 'r') as f:
        annotations = json.load(f)

    selected_frames = []
    other_candidates = {} 

    for img_data in annotations:
        file_name = img_data['file_name']
        parts = file_name.replace('.jpg', '').split('_')
        actor_id = int(parts[0].replace('Actor', ''))
        alt = parts[1].replace('a', '')
        frame_num = int(parts[2].replace('f', ''))
        
        if alt not in ['50', '70', '90']:
            continue
            
        valid_bboxes = [ann['bbox'] for ann in img_data['annotations'] if int(ann.get('visibility', 0)) >= 70]
        if not valid_bboxes:
            continue

        pose = pose_map.get(actor_id, {}).get(alt, {}).get(frame_num, "Unknown")
        is_laying = ("Laying" in pose)
        
        split_dest = 'train' if actor_id in train_actors else 'val'
        
        frame_dict = {
            'file_name': file_name,
            'bboxes': valid_bboxes,
            'img_w': img_data['width'],
            'img_h': img_data['height'],
            'split': split_dest
        }

        if alt == '50' and is_laying:
            selected_frames.append(frame_dict)
        elif alt in ['70', '90']:
            if is_laying:
                selected_frames.append(frame_dict)
            else:
                key = (actor_id, alt)
                if key not in other_candidates:
                    other_candidates[key] = []
                other_candidates[key].append(frame_dict)

    for key, frames in other_candidates.items():
        selected_frames.extend(random.sample(frames, min(5, len(frames))))

    # 4. PREPARE DIRECTORIES
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 5. GENERATE TILES (POSITIVE & BACKGROUND)
    total_positives = 0
    bg_needed_train = 0
    bg_needed_val = 0
    
    # Pass 1: Generate positive crops
    print("Generating positive 640x640 tiles...")
    crop_tasks = {'train': [], 'val': []} # Store info to extract backgrounds later
    
    for item in selected_frames:
        file_name = item['file_name']
        src_path = get_source_path(file_name, BASE_DIR)
        split = item['split']
        
        if not src_path.exists():
            continue
            
        img = None
        crop_count = 0
        
        for idx, bbox in enumerate(item['bboxes']):
            crop_res = get_safe_random_crop(bbox, item['bboxes'], item['img_w'], item['img_h'], CROP_SIZE)
            crop_coords, boxes_in_crop = crop_res
            
            if crop_coords:
                if img is None:
                    img = cv2.imread(str(src_path))
                    crop_tasks[split].append({'img_path': src_path, 'bboxes': item['bboxes'], 'img_w': item['img_w'], 'img_h': item['img_h']})
                    
                cx, cy = crop_coords
                cropped_img = img[cy:cy+CROP_SIZE, cx:cx+CROP_SIZE]
                
                new_filename = f"{file_name.replace('.jpg', '')}_crop{crop_count}"
                cv2.imwrite(str(OUTPUT_DIR / 'images' / split / f"{new_filename}.jpg"), cropped_img)
                
                with open(OUTPUT_DIR / 'labels' / split / f"{new_filename}.txt", 'w') as lf:
                    for b in boxes_in_crop:
                        yolo_line = convert_to_yolo(b, cx, cy, CROP_SIZE)
                        lf.write(f"{yolo_line}\n")
                        
                crop_count += 1
                total_positives += 1
                
                if split == 'train': bg_needed_train += 1
                else: bg_needed_val += 1

    # Pass 2: Generate 15% Background crops
    print("\nGenerating negative background tiles...")
    # Calculate required backgrounds to make them exactly 15% of the specific split
    target_bg_train = int((BG_RATIO / (1 - BG_RATIO)) * bg_needed_train)
    target_bg_val = int((BG_RATIO / (1 - BG_RATIO)) * bg_needed_val)
    
    def generate_backgrounds(split, target_bg, task_list):
        generated = 0
        random.shuffle(task_list)
        
        for task in task_list:
            if generated >= target_bg:
                break
                
            bg_coords = get_background_crop(task['bboxes'], task['img_w'], task['img_h'], CROP_SIZE)
            if bg_coords:
                img = cv2.imread(str(task['img_path']))
                cx, cy = bg_coords
                cropped_img = img[cy:cy+CROP_SIZE, cx:cx+CROP_SIZE]
                
                base_name = task['img_path'].stem
                new_filename = f"{base_name}_bg{generated}"
                
                cv2.imwrite(str(OUTPUT_DIR / 'images' / split / f"{new_filename}.jpg"), cropped_img)
                open(OUTPUT_DIR / 'labels' / split / f"{new_filename}.txt", 'w').close() # Empty txt for background
                generated += 1
                
        print(f" - {split.capitalize()}: Added {generated} background tiles.")

    generate_backgrounds('train', target_bg_train, crop_tasks['train'])
    generate_backgrounds('val', target_bg_val, crop_tasks['val'])

    # 6. CONFIG FILE
    yaml_path = OUTPUT_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n  0: human\n")

    print(f"\nDataset generation complete! Total positive tiles: {total_positives}")
    print(f"Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()