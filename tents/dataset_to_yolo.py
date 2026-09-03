DATASET_PATH = r"C:\Users\Bartek\Desktop\SUAV\tents_custom"

import os
import random
import shutil
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(DATASET_PATH)
OUTPUT_DIR = Path('tent_dataset_yolo')
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

def create_yolo_structure(base_path):
    """Creates the standard YOLO directory structure."""
    splits = ['train', 'val']
    for split in splits:
        (base_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

def get_valid_pairs(root_dir):
    """Collects all image paths and their corresponding label paths."""
    image_label_pairs = []
    
    # Dynamically find all 'imagesX' folders
    image_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith('images') and d.name[-1].isdigit()]
    
    for img_dir in image_dirs:
        # Extract the suffix (e.g., '1', '2', '6')
        suffix = img_dir.name.replace('images', '')
        lbl_dir = root_dir / f'labels{suffix}'
        
        if not lbl_dir.exists():
            print(f"Warning: Found {img_dir.name} but no matching {lbl_dir.name}. Skipping.")
            continue
            
        # Match files
        for img_path in img_dir.glob('*.*'): # Matches .png, .jpg, etc.
            if img_path.is_file():
                label_name = img_path.stem + '.txt'
                lbl_path = lbl_dir / label_name
                
                if lbl_path.exists():
                    image_label_pairs.append((img_path, lbl_path))
                else:
                    print(f"Missing label for {img_path.name}")
                    
    return image_label_pairs

def split_and_copy(pairs, output_dir):
    """Shuffles, splits, and copies files into the YOLO structure."""
    random.shuffle(pairs)
    
    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
    }
    
    for split_name, split_pairs in splits.items():
        print(f"Copying {len(split_pairs)} files to {split_name}...")
        for img_src, lbl_src in split_pairs:
            # Destination paths
            img_dst = output_dir / 'images' / split_name / img_src.name
            lbl_dst = output_dir / 'labels' / split_name / lbl_src.name
            
            # Copy files
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

if __name__ == '__main__':
    print("Preparing YOLO structure...")
    create_yolo_structure(OUTPUT_DIR)
    
    print("Gathering image-label pairs...")
    pairs = get_valid_pairs(ROOT_DIR)
    print(f"Found {len(pairs)} valid image-label pairs.")
    
    if pairs:
        split_and_copy(pairs, OUTPUT_DIR)
        print(f"Dataset successfully created in '{OUTPUT_DIR.absolute()}'")
    else:
        print("No valid pairs found. Check your directory structure.")