import os
import shutil

from matplotlib import rc

def rename_files_sequentially(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    # We ignore subdirectories to avoid errors
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Sort the files alphabetically so the numbering makes sense
    files.sort()
    
    # Iterate through the files, starting our counter at 1
    for index, filename in enumerate(files, start=1):
        
        # Extract the file extension (e.g., '.jpg', '.png')
        _, ext = os.path.splitext(filename)
        
        # Format the number to be 4 digits with leading zeros (0001, 0002, etc.)
        new_filename = f"{index:04d}{ext}"
        
        old_file_path = os.path.join(input_folder, filename)
        new_file_path = os.path.join(output_folder, new_filename)
        
        try:
            # Copy the file to the new folder with the new name
            # (shutil.copy2 preserves file metadata like creation dates)
            shutil.copy2(old_file_path, new_file_path)
            print(f"Copied and renamed: {filename} -> {new_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # --- CONFIGURE YOUR SETTINGS HERE ---
    
    INPUT_FOLDER = r"C:\Users\Bartek\Desktop\IMAV\astra_dataset"
    OUTPUT_FOLDER = r"C:\Users\Bartek\Desktop\IMAV\astra_dataset_renamed" 
    
    # ------------------------------------
    
    print(f"Reading files from: {INPUT_FOLDER}")
    rename_files_sequentially(
        input_folder=INPUT_FOLDER, 
        output_folder=OUTPUT_FOLDER
    )
    print("Done!")