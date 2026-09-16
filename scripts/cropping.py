import os
import random
import time
from pathlib import Path
import cv2

# ==========================================
# --- Configuration ---
# ==========================================
source_images_dir = r"C:\Users\Bartek\Desktop\SUAV\suas_human\human_detection_suas_clipped"
source_labels_root = r"C:\Users\Bartek\Desktop\SUAV\suas_human\human_detestion_suas_clipped_labels"
output_dataset_dir = r"C:\Users\Bartek\Desktop\SUAV\suas_human\human_detection_suas_clipped_cropped"

# Dataset settings
train_ratio = 0.8       # 80% of images for training, 20% for validation
exclude_dirs = []  # Subdirectories to ignore (e.g., thumbnails)

# Cropping & Augmentation settings
tile_size = 640
edge_margin = 0          # Margin in pixels to keep actor away from tile edges
bg_ratio_target = 0.4     # Ratio of background (empty) tiles to positive tiles
omit_probability = 0.0    # Probability (0.0 to 1.0) to randomly skip an image
random_seed = 42
# ==========================================

def collect_image_paths(root_dir: str) -> dict:
    image_paths = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for path in Path(root_dir).rglob(f"*{ext}"):
            if path.is_file() and not any(p in path.parts for p in exclude_dirs):
                image_paths[path.stem] = path
    return image_paths

def collect_label_paths(root_dir: str) -> dict:
    label_paths = {}
    # Use rglob to find .txt files anywhere inside the root directory
    for path in Path(root_dir).rglob("*.txt"):
        if path.is_file() and not any(p in path.parts for p in exclude_dirs):
            label_paths[path.stem] = path
    return label_paths

def parse_yolo_labels(label_path: Path):
    with open(label_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    
    annotations = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cls_id = parts[0]
            x_c, y_c, w, h = map(float, parts[1:5])
            annotations.append((cls_id, x_c, y_c, w, h))
    return annotations

def get_bboxes_in_crop(annotations, crop_x, crop_y, current_tile_size, img_w, img_h):
    """Finds and recalculates all bounding boxes that fall within the new crop region."""
    new_annotations = []
    crop_x_max = crop_x + current_tile_size
    crop_y_max = crop_y + current_tile_size

    for cls_id, x_c, y_c, w, h in annotations:
        bx_min = (x_c - w / 2) * img_w
        by_min = (y_c - h / 2) * img_h
        bx_max = (x_c + w / 2) * img_w
        by_max = (y_c + h / 2) * img_h

        inter_xmin = max(bx_min, crop_x)
        inter_ymin = max(by_min, crop_y)
        inter_xmax = min(bx_max, crop_x_max)
        inter_ymax = min(by_max, crop_y_max)

        if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
            new_w = (inter_xmax - inter_xmin) / current_tile_size
            new_h = (inter_ymax - inter_ymin) / current_tile_size
            new_xc = ((inter_xmin - crop_x) + (inter_xmax - inter_xmin) / 2) / current_tile_size
            new_yc = ((inter_ymin - crop_y) + (inter_ymax - inter_ymin) / 2) / current_tile_size
            new_annotations.append((cls_id, new_xc, new_yc, new_w, new_h))
            
    return new_annotations

def write_yolo_label(label_path: Path, annotations: list) -> None:
    with open(label_path, "w", encoding="utf-8") as handle:

        for cls_id, x_center, y_center, width, height in annotations:
            handle.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_split(split_name: str, split_bases: list, image_paths: dict, matched_labels: dict) -> None:
    out_img_dir = Path(output_dataset_dir) / "images" / split_name
    out_lbl_dir = Path(output_dataset_dir) / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(split_bases)
    if total_images == 0:
        print(f"\n[{split_name.upper()}] No images assigned to this split.")
        return

    print(f"\n[{split_name.upper()}] Processing {total_images} images...")

    pos_count = 0
    bg_count = 0
    skipped = {"random": 0, "empty": 0, "uncroppable": 0}
    processed_count = 0

    start_time = time.time()
    last_print_time = start_time

    for i, base_name in enumerate(split_bases, 1):
        processed_count += 1

        if random.random() < omit_probability:
            skipped["random"] += 1
            continue

        img = cv2.imread(str(image_paths[base_name]))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        if base_name not in matched_labels:
            bg_img = img[:min(tile_size, img_h), :min(tile_size, img_w)]
            bg_name = f"{base_name}_bg_unmatched.jpg"
            cv2.imwrite(str(out_img_dir / bg_name), bg_img)
            (out_lbl_dir / bg_name.replace(".jpg", ".txt")).touch()
            bg_count += 1
            continue

        annotations = parse_yolo_labels(matched_labels[base_name])
        if not annotations:
            skipped["empty"] += 1
            continue

        for cls_id, x_c, y_c, w, h in annotations:
            bx_min = int((x_c - w / 2) * img_w)
            by_min = int((y_c - h / 2) * img_h)
            bx_max = int((x_c + w / 2) * img_w)
            by_max = int((y_c + h / 2) * img_h)

            if (bx_max - bx_min) > tile_size or (by_max - by_min) > tile_size:
                skipped["uncroppable"] += 1
                continue

            min_crop_x = max(0, bx_max + edge_margin - tile_size)
            min_crop_y = max(0, by_max + edge_margin - tile_size)
            max_crop_x = min(img_w - tile_size, bx_min - edge_margin)
            max_crop_y = min(img_h - tile_size, by_min - edge_margin)

            if min_crop_x > max_crop_x:
                min_crop_x, max_crop_x = max(0, bx_max - tile_size), min(img_w - tile_size, bx_min)
            if min_crop_y > max_crop_y:
                min_crop_y, max_crop_y = max(0, by_max - tile_size), min(img_h - tile_size, by_min)
            if min_crop_x > max_crop_x or min_crop_y > max_crop_y:
                skipped["uncroppable"] += 1
                continue

            crop_x = random.randint(min_crop_x, max_crop_x)
            crop_y = random.randint(min_crop_y, max_crop_y)

            tile_img = img[crop_y:crop_y + tile_size, crop_x:crop_x + tile_size]
            tile_name = f"{base_name}_pos_{pos_count + 1}.jpg"
            cv2.imwrite(str(out_img_dir / tile_name), tile_img)

            crop_annotations = get_bboxes_in_crop(annotations, crop_x, crop_y, tile_size, img_w, img_h)
            write_yolo_label(out_lbl_dir / tile_name.replace(".jpg", ".txt"), crop_annotations)
            pos_count += 1

            target_bg_count = int(pos_count * bg_ratio_target)
            while bg_count < target_bg_count:
                valid_bg = False
                for _ in range(10):
                    bg_x = random.randint(0, max(0, img_w - tile_size))
                    bg_y = random.randint(0, max(0, img_h - tile_size))

                    # NEW LOGIC: Check the proposed background crop against ALL annotations in the image
                    # If get_bboxes_in_crop returns any boxes, it means an object is inside this crop
                    objects_in_bg = get_bboxes_in_crop(annotations, bg_x, bg_y, tile_size, img_w, img_h)
                    
                    if len(objects_in_bg) == 0:  # 100% empty of any labeled objects
                        bg_img = img[bg_y:bg_y + tile_size, bg_x:bg_x + tile_size]
                        bg_name = f"{base_name}_bg_{bg_count}.jpg"
                        cv2.imwrite(str(out_img_dir / bg_name), bg_img)
                        (out_lbl_dir / bg_name.replace(".jpg", ".txt")).touch()
                        bg_count += 1
                        valid_bg = True
                        break

                if not valid_bg:
                    break

        current_time = time.time()
        if i == total_images or i % 100 == 0 or (current_time - last_print_time) >= 1.0:
            elapsed = current_time - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            pct = (processed_count / total_images) * 100
            print(f"\r[{split_name.upper()}] {processed_count}/{total_images} ({pct:.1f}%) | "
                  f"Pos: {pos_count} | BG: {bg_count} | Skipped: {sum(skipped.values())} | {fps:.1f} img/s", 
                  end="", flush=True)
            last_print_time = current_time

    elapsed = time.time() - start_time
    print(f"\n[{split_name.upper()}] Done in {elapsed:.1f}s! Positives: {pos_count} | Backgrounds: {bg_count} | "
          f"Skipped(prob): {skipped['random']} | Skipped(empty): {skipped['empty']} | Skipped(uncroppable): {skipped['uncroppable']}")

if __name__ == "__main__":
    random.seed(random_seed)
    
    print("Starting matching and slicing process...")
    
    # 1. Gather all files
    print(f"Indexing source images from: {source_images_dir}...")
    global_image_paths = collect_image_paths(source_images_dir)
    print(f"Found {len(global_image_paths)} valid frame images.")

    print(f"Indexing source labels from: {source_labels_root}...")
    global_label_paths = collect_label_paths(source_labels_root)
    print(f"Found {len(global_label_paths)} valid label files.")

    # 2. Match Images to Labels (handles prefixes like '0aa1c2fa-img_0058')
    matched_bases = []
    matched_labels = {}
    image_stems = set(global_image_paths.keys())

    for lbl_stem, lbl_path in global_label_paths.items():
        if lbl_stem in image_stems:
            matched_bases.append(lbl_stem)
            matched_labels[lbl_stem] = lbl_path
        else:
            # Handle prefixes separated by '-' (e.g., UUID-img_name)
            if "-" in lbl_stem:
                parts = lbl_stem.split("-")
                for i in range(1, len(parts)):
                    potential_stem = "-".join(parts[i:])
                    if potential_stem in image_stems:
                        matched_bases.append(potential_stem)
                        matched_labels[potential_stem] = lbl_path
                        break
            
            # Handle prefixes separated by '_' just in case
            elif "_" in lbl_stem:
                parts = lbl_stem.split("_")
                for i in range(1, len(parts)):
                    potential_stem = "_".join(parts[i:])
                    if potential_stem in image_stems:
                        matched_bases.append(potential_stem)
                        matched_labels[potential_stem] = lbl_path
                        break

    matched_bases = sorted(set(matched_bases))
    print(f"Successfully matched {len(matched_bases)} image-label pairs.")

    unmatched_bases = image_stems - set(matched_bases)
    print(f"Images without matching labels: {len(unmatched_bases)} (will be added as backgrounds).")

    all_bases = sorted(image_stems)

    # 3. Split sequentially (First 80% to train, last 20% to val)
    # random.shuffle(matched_bases)  <-- REMOVED to ensure sequential splitting
    split_index = int(len(all_bases) * train_ratio)
    
    splits = {
        "train": all_bases[:split_index],
        "val": all_bases[split_index:]
    }

    print(f"Splitting dataset sequentially: {len(splits['train'])} for Training, {len(splits['val'])} for Validation.\n")

    # 4. Process each split
    for split_name, base_names in splits.items():
        process_split(split_name, base_names, global_image_paths, matched_labels)
        
    print("\nDataset matching and slicing complete!")