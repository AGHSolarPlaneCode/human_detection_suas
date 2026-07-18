import os
import shutil

images_source = r""
labels_source = r""
output_dataset_dir = r""

# Script for matching labels woth images Ineeded for combining files from whole nomad
VAL_ACTORS = [f"Actor{str(i).zfill(3)}" for i in range(72, 91)] # actors chosem for val 

print(f"Validation Actors Configured: {VAL_ACTORS[0]} through {VAL_ACTORS[-1]}")

# --- 2. Create YOLO Directory Structure ---
print("\n--- PHASE 1: CREATING YOLO DIRECTORIES ---")
dirs_to_make = [
    os.path.join(output_dataset_dir, 'images', 'train'),
    os.path.join(output_dataset_dir, 'images', 'val'),
    os.path.join(output_dataset_dir, 'labels', 'train'),
    os.path.join(output_dataset_dir, 'labels', 'val')
]

for d in dirs_to_make:
    os.makedirs(d, exist_ok=True)
print("Directories ready.")

print("\n--- PHASE 2: SCANNING AND MATCHING FILES ---")
image_dict = {}
label_dict = {}

for root, dirs, files in os.walk(images_source):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            base_name = os.path.splitext(file)[0]
            image_dict[base_name] = os.path.join(root, file)

for root, dirs, files in os.walk(labels_source):
    for file in files:
        if file.lower().endswith('.txt'):
            base_name = os.path.splitext(file)[0]
            label_dict[base_name] = os.path.join(root, file)

# match on name -> nomad feature images and labels are named the same
matched_bases = list(set(image_dict.keys()).intersection(set(label_dict.keys())))
matched_bases.sort() 
print(f"Found {len(matched_bases)} perfectly matched Image/Label pairs.")

print("\n--- PHASE 3: ROUTING FILES BY ACTOR (This might take a minute) ---")
train_count = 0
val_count = 0

for i, base in enumerate(matched_bases):
    src_img = image_dict[base]
    src_lbl = label_dict[base]
    img_ext = os.path.splitext(src_img)[1]
    
    is_val = any(val_actor in base for val_actor in VAL_ACTORS)
    
    if is_val:
        split_type = 'val'
        val_count += 1
    else:
        split_type = 'train'
        train_count += 1
        
    dst_img = os.path.join(output_dataset_dir, 'images', split_type, base + img_ext)
    dst_lbl = os.path.join(output_dataset_dir, 'labels', split_type, base + '.txt')
    
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)
    
    if (i + 1) % 1000 == 0:
        print(f"  -> Processed {i + 1} files...")

print(f"\nSUCCESS! Dataset built without data leakage.")
print(f"Training pairs: {train_count}")
print(f"Validation pairs: {val_count} (Exclusively Actors 72-90)")