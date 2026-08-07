import json
import os
import re

# --- Configuration ---
json_file_path = r"C:\Users\Bartek\Desktop\IMAV\mission_1\annotations.json"
output_base_dir = r"C:\Users\Bartek\Desktop\IMAV\mission_1\nomad_30m_50m_70m"
train_labels_dir = os.path.join(output_base_dir, "train", "labels")
val_labels_dir = os.path.join(output_base_dir, "val", "labels")
allowed_actors = range(1, 101)    # Actors 1 through 100
val_actor_ids = set(range(80, 100))  # Actors 80-99 go to the validation split

# Dictionary mapping altitude strings to their corresponding (min_visibility, max_visibility) percentages (0-100).
# Images with altitudes not listed here will be skipped entirely.
altitude_visibility_thresholds = {
    '30': (20, 50),  # For 30m height, require between 20% and 50% visibility
    '50': (50, 100),  # For 50m height, require between 50% and 100% visibility
    '70': (100, 100)  # For 70m height, require exactly 100% visibility
}

# Create the output directories if they don't exist
for labels_dir in (train_labels_dir, val_labels_dir):
    os.makedirs(labels_dir, exist_ok=True)

# Load the JSON data
print(f"Loading {json_file_path}...")
with open(json_file_path, 'r') as f:
    data = json.load(f)

print("Processing annotations with Actor, Altitude, and dynamic Visibility filters...")
processed_images_count = 0
train_images_count = 0
val_images_count = 0
skipped_actor_count = 0
skipped_altitude_count = 0
skipped_boxes_visibility_count = 0  # Counter for filtered bounding boxes

for image_data in data:
    file_name = image_data['file_name']
    
    # 1. Extract and Filter by Actor ID (e.g., "Actor095" -> 95)
    actor_match = re.search(r'Actor(\d{3})', file_name)
    if not actor_match:
        continue
    
    actor_id = int(actor_match.group(1))
    if actor_id not in allowed_actors:
        skipped_actor_count += 1
        continue
        
    # 2. Extract and Filter by Altitude (e.g., "_a50_" -> "50")
    alt_match = re.search(r'_a(\d+)_', file_name)
    if not alt_match:
        continue
        
    altitude = alt_match.group(1)
    
    # Check if the extracted altitude is in our configuration dictionary
    if altitude not in altitude_visibility_thresholds:
        skipped_altitude_count += 1
        continue
    
    # Get the specific minimum and maximum visibility required for this altitude
    current_min_visibility, current_max_visibility = altitude_visibility_thresholds[altitude]
        
    # If it passes both image-level filters, process the bounding boxes
    img_width = image_data['width']
    img_height = image_data['height']
    annotations = image_data.get('annotations', [])
    
    # Split the labels between train and validation folders
    split_name = 'val' if actor_id in val_actor_ids else 'train'
    output_labels_dir = val_labels_dir if split_name == 'val' else train_labels_dir

    # Create a corresponding .txt filename
    txt_filename = os.path.splitext(file_name)[0] + '.txt'
    txt_filepath = os.path.join(output_labels_dir, txt_filename)
    
    valid_boxes_found = False
    
    # Open the text file and write the normalized YOLO coordinates
    with open(txt_filepath, 'w') as txt_file:
        for ann in annotations:
            # 3. Filter by Visibility (using the thresholds specific to this altitude)
            # Visibility is a string in your JSON (e.g., "100"), cast to int. Default to 100 if missing.
            visibility = int(ann.get('visibility', '100'))
            if visibility < current_min_visibility or visibility > current_max_visibility:
                skipped_boxes_visibility_count += 1
                continue  # Skip this specific bounding box
                
            # NOMAD format: [x_min, y_min, box_width, box_height]
            x_min, y_min, box_w, box_h = ann['bbox']
            category_id = ann['category_id'] # 0 for person
            
            # YOLO Math: Calculate center points and normalize everything between 0 and 1
            x_center = (x_min + (box_w / 2.0)) / img_width
            y_center = (y_min + (box_h / 2.0)) / img_height
            norm_width = box_w / img_width
            norm_height = box_h / img_height
            
            # Ensure values are strictly between 0 and 1 (clamps edge cases)
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))
            
            # Write line: <class> <x_center> <y_center> <width> <height>
            txt_file.write(f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            valid_boxes_found = True
            
    # Optional clean-up: if an image had boxes but ALL were filtered out due to low visibility, 
    # it leaves an empty .txt file. YOLO handles empty .txt files fine (treats as background images),
    # but we still count the image as processed.
    processed_images_count += 1
    if split_name == 'val':
        val_images_count += 1
    else:
        train_images_count += 1

print(f"Done! Successfully generated {processed_images_count} YOLO label files.")
print(f"Train label files written to: {train_labels_dir}")
print(f"Validation label files written to: {val_labels_dir}")
print(f"Train images processed: {train_images_count}")
print(f"Validation images processed: {val_images_count}")
print(f"Skipped {skipped_actor_count} images (Actors outside 1-100).")
print(f"Skipped {skipped_altitude_count} images (Altitude not in configured list).")
print(f"Filtered out {skipped_boxes_visibility_count} bounding boxes outside the configured min/max visibility thresholds.")