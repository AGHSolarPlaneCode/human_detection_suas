import os
import random
from pathlib import Path

# --- Configuration ---
DATASET_DIR = r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\four_class_dataset"
TARGET_BG_RATIO = 0.15  # Backgrounds will make up 15% of the final dataset

def balance_backgrounds():
    dataset_path = Path(DATASET_DIR)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = dataset_path / 'images' / split
        lbl_dir = dataset_path / 'labels' / split
        
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Skipping '{split}': Directory not found.")
            continue
            
        print(f"\nProcessing '{split}' split...")
        
        # Gather all images
        all_images = list(img_dir.glob('*.[jJ][pP][gG]'))
        
        # Separate into object and background lists based on the naming convention "_bg_"
        obj_images = [img for img in all_images if '_bg_' not in img.stem]
        bg_images = [img for img in all_images if '_bg_' in img.stem]
        
        obj_count = len(obj_images)
        bg_count = len(bg_images)
        
        print(f"  Current: {obj_count} objects, {bg_count} backgrounds")
        
        if obj_count == 0:
            print("  No objects found. Skipping.")
            continue
            
        # Calculate target background count
        # If BG is 15% of total, then OBJ is 85% of total. 
        # So Target_BG = (0.15 / 0.85) * OBJ
        target_bg_count = int((TARGET_BG_RATIO / (1.0 - TARGET_BG_RATIO)) * obj_count)
        
        if bg_count <= target_bg_count:
            print(f"  Background count is already at or below target ({target_bg_count}). No deletion needed.")
            continue
            
        # Calculate how many to delete and pick randomly
        delete_count = bg_count - target_bg_count
        print(f"  Target backgrounds: {target_bg_count}. Deleting {delete_count} files...")
        
        # Randomly select which background files to delete
        bgs_to_delete = random.sample(bg_images, delete_count)
        
        deleted_imgs = 0
        deleted_lbls = 0
        
        for bg_img in bgs_to_delete:
            # 1. Delete Image
            try:
                bg_img.unlink()
                deleted_imgs += 1
            except Exception as e:
                print(f"  Error deleting {bg_img.name}: {e}")
                
            # 2. Delete corresponding label
            lbl_file = lbl_dir / f"{bg_img.stem}.txt"
            if lbl_file.exists():
                try:
                    lbl_file.unlink()
                    deleted_lbls += 1
                except Exception as e:
                    print(f"  Error deleting {lbl_file.name}: {e}")
                    
        print(f"  Done. Deleted {deleted_imgs} images and {deleted_lbls} label files.")

if __name__ == "__main__":
    # Optional: set a fixed seed if you want the random deletion to be reproducible
    # random.seed(42) 
    balance_backgrounds()