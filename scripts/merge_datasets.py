import os
import shutil
from pathlib import Path

# ==========================================
# --- Configuration ---
# ==========================================
# List the paths to the root folders of the datasets you want to merge
input_datasets = [
    r"C:\Users\Bartek\Desktop\SUAV\generated\images_human3_cropped",
    r"C:\Users\Bartek\Desktop\SUAV\generated\human1",
    r"C:\Users\Bartek\Desktop\SUAV\generated\human2"
]

# The directory where the final merged dataset will be saved
output_dataset_dir = r"C:\Users\Bartek\Desktop\SUAV\generated\merged_v2"

# The splits you expect to find in your datasets
splits = ["train", "val"]

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png"}
# ==========================================

def merge_datasets(input_dirs: list, output_dir: str, splits: list):
    out_root = Path(output_dir)
    
    # Create the base output directories
    for split in splits:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images_copied = 0
    total_labels_copied = 0

    print(f"Starting merge into: {out_root}\n")

    for ds_idx, ds_path_str in enumerate(input_dirs):
        ds_path = Path(ds_path_str)
        
        if not ds_path.exists():
            print(f"⚠️ Warning: Dataset not found -> {ds_path}. Skipping.")
            continue

        # Use the folder name as a unique prefix to prevent filename collisions.
        # Fallback to ds_idx if somehow there's an issue with the name.
        prefix = ds_path.name if ds_path.name else f"ds{ds_idx}"
        print(f"📦 Processing Dataset: '{ds_path.name}' (Prefix: {prefix}_)")

        for split in splits:
            img_dir = ds_path / "images" / split
            lbl_dir = ds_path / "labels" / split

            if not img_dir.exists():
                continue

            images_in_split = 0
            
            # Iterate through all files in the images folder
            for img_file in img_dir.rglob("*"):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    
                    # Construct new names with the prefix to avoid overwriting
                    new_img_name = f"{prefix}_{img_file.name}"
                    new_lbl_name = f"{prefix}_{img_file.stem}.txt"

                    dest_img_path = out_root / "images" / split / new_img_name
                    dest_lbl_path = out_root / "labels" / split / new_lbl_name

                    # 1. Copy the image
                    shutil.copy2(img_file, dest_img_path)
                    images_in_split += 1
                    total_images_copied += 1

                    # 2. Look for the corresponding label
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    
                    if lbl_file.exists():
                        # Copy the label if it exists
                        shutil.copy2(lbl_file, dest_lbl_path)
                        total_labels_copied += 1
                    else:
                        # If it's a background image, it might not have a label file, 
                        # or it might have an empty one. We create an empty label to be safe.
                        dest_lbl_path.touch()
            
            if images_in_split > 0:
                print(f"  └─ [{split.upper()}] Copied {images_in_split} images.")

    print("\n✅ Merge Complete!")
    print(f"Total Images Copied: {total_images_copied}")
    print(f"Total Labels Processed: {total_labels_copied}")


if __name__ == "__main__":
    merge_datasets(input_datasets, output_dataset_dir, splits)