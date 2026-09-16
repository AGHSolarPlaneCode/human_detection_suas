import os
from pathlib import Path

def process_backgrounds(root_folder):
    # Set of valid image extensions to ensure we don't rename non-image files (like hidden OS files or scripts)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    
    root_path = Path(root_folder)
    
    # Counter for the sequential naming
    counter = 1
    
    # .rglob('*') makes the search recursive through all subdirectories
    for filepath in root_path.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
            parent_dir = filepath.parent
            ext = filepath.suffix
            
            # Define the new names
            new_image_name = f"bg_{counter}{ext}"
            new_label_name = f"bg_{counter}.txt"
            
            new_image_path = parent_dir / new_image_name
            new_label_path = parent_dir / new_label_name
            
            # 1. Rename the original background file
            filepath.rename(new_image_path)
            
            # 2. Create the empty label file with the same base name
            new_label_path.touch()
            
            print(f"Renamed: {filepath.name} -> {new_image_name} | Created: {new_label_name}")
            
            counter += 1

if __name__ == "__main__":
    # Prompt the user for the target folder
    target_directory = r"c:\Users\Bartek\Desktop\SUAV\backgrounds\bg_dataset_testy"
    
    # Verify the directory exists before running
    if os.path.exists(target_directory) and os.path.isdir(target_directory):
        print(f"Scanning '{target_directory}' recursively...")
        process_backgrounds(target_directory)
        print("\nFinished processing all backgrounds!")
    else:
        print("Error: The specified folder does not exist.")