import shutil
from pathlib import Path

# --- Configuration ---
# Update these paths to match the locations on your computer
IMAGES_BASE_DIR = r"C:\Users\Bartek\Desktop\SUAV\testy_dataset\images"  # Folder containing train/, val/, test/
LABELS_SOURCE_DIR = r"C:\Users\Bartek\Desktop\SUAV\testy_dataset\labels" # Folder currently containing all .txt files

def organize_labels():
    images_base = Path(IMAGES_BASE_DIR)
    labels_source = Path(LABELS_SOURCE_DIR)
    
    # Define the splits we are looking for
    splits = ['train', 'val', 'test']
    
    moved_count = 0
    missing_count = 0

    for split in splits:
        img_split_dir = images_base / split
        
        # Check if the image split folder (e.g., images/train) actually exists
        if not img_split_dir.exists():
            print(f"Skipping '{split}': Directory {img_split_dir} not found.")
            continue
            
        # Create the corresponding labels split directory (e.g., labels/train)
        label_split_dir = labels_source / split
        label_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Go through every image in the current split folder
        for img_path in img_split_dir.iterdir():
            if img_path.is_file():
                # Get the filename without the extension (e.g., 'image_01' from 'image_01.jpg')
                # and append '.txt'
                label_filename = img_path.stem + '.txt'
                
                source_label_path = labels_source / label_filename
                dest_label_path = label_split_dir / label_filename
                
                # Move the file if we find it in the main labels folder
                if source_label_path.exists():
                    shutil.move(str(source_label_path), str(dest_label_path))
                    moved_count += 1
                elif not dest_label_path.exists():
                    # If it's not in the source AND not already moved, log it
                    print(f"Warning: No label found for {img_path.name}")
                    missing_count += 1

    print("\n--- Summary ---")
    print(f"Successfully moved {moved_count} label files.")
    if missing_count > 0:
        print(f"Could not find labels for {missing_count} images.")

if __name__ == "__main__":
    organize_labels()