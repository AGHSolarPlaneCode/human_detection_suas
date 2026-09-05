import os
from PIL import Image

def slice_images(input_folder, output_folder, tile_size=640, drop_partials=True):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Common image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(input_folder, filename)
        
        try:
            # Open the image
            with Image.open(img_path) as img:
                width, height = img.size
                base_name, ext = os.path.splitext(filename)
                
                # Iterate through the image in steps of tile_size
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        
                        # Define the coordinates for the crop box
                        right = x + tile_size
                        bottom = y + tile_size
                        
                        # If drop_partials is True, skip edge tiles smaller than 640x640
                        if drop_partials and (right > width or bottom > height):
                            continue
                            
                        # Crop the tile
                        box = (x, y, min(right, width), min(bottom, height))
                        tile = img.crop(box)
                        
                        # Save the tile with its coordinates in the filename
                        tile_filename = f"{base_name}_x{x}_y{y}{ext}"
                        tile_path = os.path.join(output_folder, tile_filename)
                        tile.save(tile_path)
                        
            print(f"Successfully sliced: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # --- CONFIGURE YOUR SETTINGS HERE ---
    
    INPUT_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\flight_data\bg"
    OUTPUT_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\flight_data\bg_sliced"
    
    TILE_SIZE = 640
    
    # Set to False if you want to keep the smaller leftover edge pieces
    DROP_PARTIALS = True 
    
    # ------------------------------------
    
    print(f"Reading images from: {INPUT_FOLDER}")
    slice_images(
        input_folder=INPUT_FOLDER, 
        output_folder=OUTPUT_FOLDER, 
        tile_size=TILE_SIZE,
        drop_partials=DROP_PARTIALS
    )
    print("Done!")