import shutil
from pathlib import Path

# 1. Define your folders
source_folder = Path(r"C:\Users\Bartek\Desktop\SUAV\whole_nomad\TN")
destination_folder = Path(r"C:\Users\Bartek\Desktop\SUAV\whole_nomad\TN_a50")

# 2. Create the destination folder if it doesn't already exist
destination_folder.mkdir(parents=True, exist_ok=True)

# 3. Find and copy the files
count = 0
print("Starting file copy...\n")

# .glob() looks for any .jpg file with 'a50' anywhere in the name
for file_path in source_folder.glob("*a50*.jpg"):
    # Create the full path for where the file will be saved
    dest_path = destination_folder / file_path.name
    
    # copy2 copies the file and preserves the original creation/modification dates
    shutil.copy2(file_path, dest_path)
    
    print(f"Copied: {file_path.name}")
    count += 1

print(f"\nDone! Successfully copied {count} files.")