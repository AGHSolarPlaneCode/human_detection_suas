import os
from pathlib import Path

# --- Configuration ---

# Define the mapping from old class ID to new class ID
# Format: {old_class_id: new_class_id}
CLASS_MAPPING = {
    0: 3,  # Example: Change old class 0 to new class 1
}

# If True, bounding boxes with classes NOT listed in CLASS_MAPPING will be deleted.
# If False, unmapped classes will be kept with their original ID.
DROP_UNMAPPED_CLASSES = False

# Paths
INPUT_LABELS_DIR = r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\tent_dataset_yolo\labels"   # Can be the root folder containing train/, val/, test/
OUTPUT_LABELS_DIR = r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\tent_dataset_yolo\labels_remap" # Safer to output to a new folder to avoid accidental overwrites

def remap_yolo_labels():
    in_dir = Path(INPUT_LABELS_DIR)
    out_dir = Path(OUTPUT_LABELS_DIR)
    
    if not in_dir.exists():
        print(f"Error: Input directory '{in_dir}' not found.")
        return

    # Find all .txt files recursively (this easily handles train/val subfolders)
    txt_files = list(in_dir.rglob('*.txt'))
    print(f"Found {len(txt_files)} label files. Processing...")

    processed_files_count = 0
    deleted_boxes_count = 0
    changed_boxes_count = 0

    for txt_path in txt_files:
        # Recreate the exact folder structure in the output directory
        rel_path = txt_path.relative_to(in_dir)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # Check if line has YOLO format: class cx cy w h
            if len(parts) >= 5: 
                old_class = int(parts[0])
                
                if old_class in CLASS_MAPPING:
                    new_class = CLASS_MAPPING[old_class]
                    # Rebuild the line with the new class and original coordinates
                    new_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
                    changed_boxes_count += 1
                elif not DROP_UNMAPPED_CLASSES:
                    # Keep the original line if we aren't dropping unmapped classes
                    new_lines.append(line) 
                else:
                    # Drop it
                    deleted_boxes_count += 1
                    
        # Write to the new output file
        with open(out_path, 'w') as f:
            f.writelines(new_lines)
            
        processed_files_count += 1
        
    print("\n--- Remapping Complete ---")
    print(f"Processed {processed_files_count} files.")
    print(f"Successfully changed {changed_boxes_count} bounding boxes.")
    if DROP_UNMAPPED_CLASSES:
        print(f"Dropped {deleted_boxes_count} unmapped bounding boxes.")
    print(f"Saved mapped labels to: {out_dir.absolute()}")

if __name__ == "__main__":
    remap_yolo_labels()