import os
import random
import time
from pathlib import Path

import cv2

# --- Configuration ---
source_images_dir = r"C:\Users\Bartek\Desktop\SUAV\whole_nomad"
source_labels_root = r"C:\Users\Bartek\Desktop\IMAV\mission_1\nomad_30m_50m_70m"
output_dataset_dir = r"C:\Users\Bartek\Desktop\IMAV\mission_1\crop_30_50_70"

tile_size = 640
edge_margin = 32  # Margin in pixels to keep actor away from tile edges
bg_ratio_target = 0.2
omit_probability = 0.5  # 50% chance to omit any image
splits = ["train", "val"]


def collect_image_paths(root_dir: str) -> dict:
    image_paths = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for path in Path(root_dir).rglob(f"*{ext}"):
            # Exclude keyframe/thumbnail folders (TN, TN_a50)
            if path.is_file() and not any(p in path.parts for p in ("TN", "TN_a50")):
                image_paths[path.stem] = path
    return image_paths


def collect_label_paths(labels_dir: str) -> dict:
    label_paths = {}
    for path in Path(labels_dir).glob("*.txt"):
        if path.is_file():
            label_paths[path.stem] = path
    return label_paths


def parse_yolo_labels(label_path: Path):
    with open(label_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    annotations = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = parts[0]
        x_c, y_c, w, h = map(float, parts[1:5])
        annotations.append((cls_id, x_c, y_c, w, h))
    return annotations


def write_yolo_label(label_path: Path, cls_id: str, x_center: float, y_center: float, width: float, height: float) -> None:
    with open(label_path, "w", encoding="utf-8") as handle:
        handle.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_split(split_name: str, image_paths: dict) -> None:
    labels_dir = Path(source_labels_root) / split_name / "labels"
    out_img_dir = Path(output_dataset_dir) / "images" / split_name
    out_lbl_dir = Path(output_dataset_dir) / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_paths = collect_label_paths(str(labels_dir))
    matched_bases = sorted(set(image_paths.keys()) & set(label_paths.keys()))
    total_matched = len(matched_bases)

    print(f"\n[{split_name.upper()}] Processing {total_matched} matched image-label pairs...")

    pos_count = 0
    bg_count = 0
    skipped_random = 0
    skipped_empty = 0
    skipped_uncroppable = 0
    processed_count = 0

    start_time = time.time()
    last_print_time = start_time

    for i, base_name in enumerate(matched_bases, 1):
        processed_count += 1

        # 1. 50% random chance to omit image
        if random.random() < omit_probability:
            skipped_random += 1
            continue

        lbl_path = label_paths[base_name]
        annotations = parse_yolo_labels(lbl_path)

        # 2. Omit image if label is empty
        if not annotations:
            skipped_empty += 1
            continue

        # Load image only if it passes omission checks
        img_path = image_paths[base_name]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        for cls_id, x_c, y_c, w, h in annotations:
            bx_min = int((x_c - w / 2) * img_w)
            by_min = int((y_c - h / 2) * img_h)
            bx_max = int((x_c + w / 2) * img_w)
            by_max = int((y_c + h / 2) * img_h)

            if (bx_max - bx_min) > tile_size or (by_max - by_min) > tile_size:
                skipped_uncroppable += 1
                continue

            # Try cropping with safety margin to prevent actor from being right at tile edge
            min_crop_x = max(0, bx_max + edge_margin - tile_size)
            min_crop_y = max(0, by_max + edge_margin - tile_size)
            max_crop_x = min(img_w - tile_size, bx_min - edge_margin)
            max_crop_y = min(img_h - tile_size, by_min - edge_margin)

            # Fallback if margin constraints are too tight near image borders
            if min_crop_x > max_crop_x:
                min_crop_x = max(0, bx_max - tile_size)
                max_crop_x = min(img_w - tile_size, bx_min)

            if min_crop_y > max_crop_y:
                min_crop_y = max(0, by_max - tile_size)
                max_crop_y = min(img_h - tile_size, by_min)

            if min_crop_x > max_crop_x or min_crop_y > max_crop_y:
                skipped_uncroppable += 1
                continue

            crop_x = random.randint(min_crop_x, max_crop_x)
            crop_y = random.randint(min_crop_y, max_crop_y)

            tile_img = img[crop_y:crop_y + tile_size, crop_x:crop_x + tile_size]
            tile_name = f"{base_name}_pos_{pos_count + 1}.jpg"
            cv2.imwrite(str(out_img_dir / tile_name), tile_img)

            new_x_min = bx_min - crop_x
            new_y_min = by_min - crop_y
            new_x_max = bx_max - crop_x
            new_y_max = by_max - crop_y

            new_w = (new_x_max - new_x_min) / tile_size
            new_h = (new_y_max - new_y_min) / tile_size
            new_xc = (new_x_min / tile_size) + (new_w / 2)
            new_yc = (new_y_min / tile_size) + (new_h / 2)

            write_yolo_label(
                out_lbl_dir / tile_name.replace(".jpg", ".txt"),
                cls_id,
                new_xc,
                new_yc,
                new_w,
                new_h,
            )
            pos_count += 1

            target_bg_count = int(pos_count * bg_ratio_target)
            while bg_count < target_bg_count:
                valid_bg = False
                for _ in range(10):
                    bg_x = random.randint(0, max(0, img_w - tile_size))
                    bg_y = random.randint(0, max(0, img_h - tile_size))

                    overlap = not (
                        bx_max <= bg_x
                        or bx_min >= bg_x + tile_size
                        or by_max <= bg_y
                        or by_min >= bg_y + tile_size
                    )

                    if not overlap:
                        bg_img = img[bg_y:bg_y + tile_size, bg_x:bg_x + tile_size]
                        bg_name = f"{base_name}_bg_{bg_count}.jpg"
                        cv2.imwrite(str(out_img_dir / bg_name), bg_img)
                        (out_lbl_dir / bg_name.replace(".jpg", ".txt")).touch()
                        bg_count += 1
                        valid_bg = True
                        break

                if not valid_bg:
                    break

        # Progress reporting every 100 images or every second
        current_time = time.time()
        if i == total_matched or i % 100 == 0 or (current_time - last_print_time) >= 1.0:
            elapsed = current_time - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            pct = (processed_count / total_matched) * 100
            print(
                f"\r[{split_name.upper()}] {processed_count}/{total_matched} ({pct:.1f}%) | "
                f"Pos: {pos_count} | BG: {bg_count} | Skip(50%): {skipped_random} | "
                f"Skip(Empty): {skipped_empty} | {fps:.1f} img/s",
                end="",
                flush=True,
            )
            last_print_time = current_time

    elapsed = time.time() - start_time
    print(
        f"\n[{split_name.upper()}] Done in {elapsed:.1f}s! Positives: {pos_count} | "
        f"Backgrounds: {bg_count} | Skipped (50% rule): {skipped_random} | "
        f"Skipped (Empty label): {skipped_empty}"
    )


if __name__ == "__main__":
    random.seed(42)
    print("Starting matching and slicing process...")
    print(f"Indexing source images from: {source_images_dir}...")
    global_image_paths = collect_image_paths(source_images_dir)
    print(f"Found {len(global_image_paths)} valid frame images.")

    for split in splits:
        process_split(split, global_image_paths)
    print("\nDataset matching and slicing complete!")


