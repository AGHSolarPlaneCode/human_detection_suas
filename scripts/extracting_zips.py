import os
import zipfile
# for extracting many zip files, I needed when I downloaded as zips labels for 90 actors and I had 180 zips to exctract
source_folder = r"path_to_your_source_folder" 
main_destination_folder = r"path_to_your_main_destination_folder"

os.makedirs(main_destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):

    if filename.endswith(".zip"):
        
        zip_path = os.path.join(source_folder, filename)
        
        folder_name = filename[:-4] 
        specific_extraction_path = os.path.join(main_destination_folder, folder_name)
        
        os.makedirs(specific_extraction_path, exist_ok=True)
        
        print(f"Extracting {filename} into \\{folder_name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(specific_extraction_path)
        except zipfile.BadZipFile:
            print(f"  -> ERROR: {filename} is corrupted or not a valid zip file. Skipping.")

print("\nAll done! Your files are extracted and organized.")