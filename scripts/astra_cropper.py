import os
import cv2
import numpy as np
import random
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = r"C:\Users\Bartek\Desktop\IMAV\astra_dataset"         # Your input dataset folder
OUTPUT_DIR = r"C:\Users\Bartek\Desktop\IMAV\dataset_cropped"  # Where the cropped dataset will be saved

# Fixed crop size in pixels
CROP_WIDTH = 640  
CROP_HEIGHT = 640 

# How many 640x640 tiles to extract per original image
CROPS_PER_IMAGE = 6 

# The threshold for trying a different slice. 
# If a bounding box is cut and retains less than 10% of its original area, we retry.
MIN_AREA_RATIO = 0.10 
MAX_RETRIES = 50 # How many times to try shifting the slice before giving up
# ==========================================

def read_yolo_labels(label_path, img_w, img_h):
    """Reads Ultralytics OBB labels as [class, [(x, y), ...]]."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            points = [
                (float(parts[i]) * img_w, float(parts[i + 1]) * img_h)
                for i in range(1, 9, 2)
            ]
            boxes.append([class_id, points])
    return boxes

def clip_polygon(points, start_x, start_y, end_x, end_y):
    """Clips a polygon to the crop rectangle using Sutherland-Hodgman clipping."""
    polygon = points
    for edge in ('left', 'right', 'top', 'bottom'):
        if not polygon:
            break
        clipped = []
        for current, previous in zip(polygon, polygon[-1:] + polygon[:-1]):
            if edge == 'left':
                current_inside = current[0] >= start_x
                previous_inside = previous[0] >= start_x
            elif edge == 'right':
                current_inside = current[0] <= end_x
                previous_inside = previous[0] <= end_x
            elif edge == 'top':
                current_inside = current[1] >= start_y
                previous_inside = previous[1] >= start_y
            else:
                current_inside = current[1] <= end_y
                previous_inside = previous[1] <= end_y

            if current_inside != previous_inside:
                dx = current[0] - previous[0]
                dy = current[1] - previous[1]
                if edge in ('left', 'right'):
                    x = start_x if edge == 'left' else end_x
                    y = previous[1] + dy * (x - previous[0]) / dx if dx else previous[1]
                else:
                    y = start_y if edge == 'top' else end_y
                    x = previous[0] + dx * (y - previous[1]) / dy if dy else previous[0]
                clipped.append((x, y))
            if current_inside:
                clipped.append(current)
        polygon = clipped
    return polygon

def get_valid_crop(img_w, img_h, crop_w, crop_h, boxes):
    """
    Tries to find a valid slice where no bounding box is cut so severely 
    that it has less than MIN_AREA_RATIO (10%) of its area left.
    """
    for _ in range(MAX_RETRIES):
        start_x = random.randint(0, img_w - crop_w)
        start_y = random.randint(0, img_h - crop_h)
        slice_coords = (start_x, start_y, start_x + crop_w, start_y + crop_h)
        
        valid_slice = True
        new_boxes = []
        
        for box in boxes:
            class_id, points = box
            orig_area = abs(cv2.contourArea(np.array(points, dtype='float32')))
            if orig_area <= 0:
                continue
            clipped_points = clip_polygon(
                points, start_x, start_y, start_x + crop_w, start_y + crop_h
            )
            if len(clipped_points) < 3:
                continue

            inter_area = abs(cv2.contourArea(np.array(clipped_points, dtype='float32')))
            if inter_area / orig_area < MIN_AREA_RATIO:
                valid_slice = False
                break

            rectangle = cv2.boxPoints(cv2.minAreaRect(
                np.array(clipped_points, dtype='float32')
            ))
            shifted_points = [
                (float(x - start_x), float(y - start_y))
                for x, y in rectangle
            ]
            new_boxes.append([class_id, shifted_points])
                    
        if valid_slice and new_boxes:
            return slice_coords, new_boxes
            
    # If it fails to find a valid slice after MAX_RETRIES, return None
    return None, None

def save_yolo_labels(out_path, boxes, crop_w, crop_h):
    """Converts OBB corner points back to normalized YOLO format and saves."""
    with open(out_path, 'w') as f:
        for box in boxes:
            class_id, points = box
            normalized = ' '.join(
                f'{x / crop_w:.6f} {y / crop_h:.6f}'
                for x, y in points
            )
            f.write(f"{class_id} {normalized}\n")

def process_dataset(split):
    """Processes either the 'train' or 'val' split"""
    img_dir_in = Path(INPUT_DIR) / 'images' / split
    lbl_dir_in = Path(INPUT_DIR) / 'labels' / split
    
    img_dir_out = Path(OUTPUT_DIR) / 'images' / split
    lbl_dir_out = Path(OUTPUT_DIR) / 'labels' / split
    
    # Create output directories
    img_dir_out.mkdir(parents=True, exist_ok=True)
    lbl_dir_out.mkdir(parents=True, exist_ok=True)
    
    if not img_dir_in.exists():
        print(f"Directory {img_dir_in} not found. Skipping {split} split.")
        return

    images = [f for f in os.listdir(img_dir_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing {split} set ({len(images)} images)...")
    for img_name in images:
        img_path = img_dir_in / img_name
        lbl_path = lbl_dir_in / (img_path.stem + ".txt")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        orig_h, orig_w = image.shape[:2]
        
        # Read absolute bounding box coords based on original image size
        boxes = read_yolo_labels(str(lbl_path), orig_w, orig_h)
        
        crop_w = CROP_WIDTH
        crop_h = CROP_HEIGHT
        
        # Pad the image if it's smaller than the target crop size
        if orig_w < crop_w or orig_h < crop_h:
            pad_right = max(0, crop_w - orig_w)
            pad_bottom = max(0, crop_h - orig_h)
            image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
        img_h, img_w = image.shape[:2] # Update sizes after potential padding
        
        for i in range(CROPS_PER_IMAGE):
            slice_coords, new_boxes = get_valid_crop(img_w, img_h, crop_w, crop_h, boxes)
            
            if slice_coords is not None:
                start_x, start_y, end_x, end_y = slice_coords
                cropped_img = image[start_y:end_y, start_x:end_x]
                
                # Naming format: originalname_crop0.jpg
                out_img_name = f"{img_path.stem}_crop{i}{img_path.suffix}"
                out_lbl_name = f"{img_path.stem}_crop{i}.txt"
                
                out_img_path = img_dir_out / out_img_name
                out_lbl_path = lbl_dir_out / out_lbl_name
                
                cv2.imwrite(str(out_img_path), cropped_img)
                save_yolo_labels(str(out_lbl_path), new_boxes, crop_w, crop_h)
            else:
                print(f"Could not find a valid 640x640 slice for {img_name} on attempt {i+1} after {MAX_RETRIES} retries.")

if __name__ == "__main__":
    process_dataset('train')
    process_dataset('val')
    print("\nDone! Dataset cropped to 640x640 tiles successfully.")