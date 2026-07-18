import os
import cv2
import random
# whole paths, do for train and val separately
input_images_dir = r""
input_labels_dir = r""
out_images_dir = r""
out_labels_dir = r""
SLICE_SIZE = 512 # size in pixels 
EMPTY_KEEP_PROB = 0.01 # the chance of keeping backgorund tile with no human, for 1% you got ~15% of backgrounds

os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)

print(f"Checking for images in: {os.path.abspath(input_images_dir)}")
if not os.path.exists(input_images_dir):
    print("ERROR: The input_images_dir does not exist! Please check your paths.")
    exit()

all_files = os.listdir(input_images_dir)
print(f"Found {len(all_files)} total files in the input directory.")

processed_images = 0
saved_tiles = 0

for img_name in all_files:
    if not img_name.lower().endswith(('.jpg', '.jpeg')): 
        continue
        
    img_path = os.path.join(input_images_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(input_labels_dir, base_name + '.txt')
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"WARNING: OpenCV could not read image -> {img_name}. Is it corrupted?")
        continue
        
    img_h, img_w = img.shape[:2]
    
    boxes = []
    # we have to calculate where the label box is
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_c, y_c, w, h = [float(v) for v in parts[1:]]
                    abs_w, abs_h = w * img_w, h * img_h
                    abs_x = (x_c * img_w) - (abs_w / 2)
                    abs_y = (y_c * img_h) - (abs_h / 2)
                    boxes.append([class_id, abs_x, abs_y, abs_w, abs_h])
    else:
        print(f"WARNING: No matching label file found for {img_name}")

    # sliding window logic -> can lead to cutting people in half -> potential problem
    # genrally this slices and calculated new labels 
    for y in range(0, img_h, SLICE_SIZE):
        for x in range(0, img_w, SLICE_SIZE):
            x1, y1 = x, y
            x2 = min(x + SLICE_SIZE, img_w)
            y2 = min(y + SLICE_SIZE, img_h)
            
            if (x2 - x1) < 500 or (y2 - y1) < 500: continue
            
            crop_img = img[y1:y2, x1:x2]
            crop_h, crop_w = crop_img.shape[:2]
            
            crop_boxes = []
            for box in boxes:
                cls_id, bx, by, bw, bh = box
                if (bx < x2 and bx + bw > x1 and by < y2 and by + bh > y1):
                    new_bx = max(0, bx - x1)
                    new_by = max(0, by - y1)
                    new_bw = min(crop_w, (bx + bw) - x1) - new_bx
                    new_bh = min(crop_h, (by + bh) - y1) - new_by
                    
                    new_xc = (new_bx + new_bw / 2) / crop_w
                    new_yc = (new_by + new_bh / 2) / crop_h
                    norm_w = new_bw / crop_w
                    norm_h = new_bh / crop_h
                    
                    crop_boxes.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            tile_name = f"{base_name}_x{x}_y{y}"
            
            if len(crop_boxes) > 0:
                cv2.imwrite(os.path.join(out_images_dir, tile_name + '.jpg'), crop_img)
                with open(os.path.join(out_labels_dir, tile_name + '.txt'), 'w') as f:
                    f.write('\n'.join(crop_boxes))
                saved_tiles += 1
            else:
                if random.random() < EMPTY_KEEP_PROB:
                    cv2.imwrite(os.path.join(out_images_dir, tile_name + '.jpg'), crop_img)
                    saved_tiles += 1
                    
    processed_images += 1
    if processed_images % 50 == 0:
        print(f"Processed {processed_images} images... generated {saved_tiles} tiles so far.")

print(f"\nSUCCESS! Processed {processed_images} massive images and created {saved_tiles} sliced tiles.")