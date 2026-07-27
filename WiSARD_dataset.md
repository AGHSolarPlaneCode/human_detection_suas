# WiSARD Dataset: Usage and Preprocessing Guide

## Download

Dataset link:
https://drive.google.com/file/d/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht/view?usp=sharing

## Dataset Structure (after unzip)

- The dataset root contains multiple subfolders.
- Each subfolder typically corresponds to one collection event (for example, a drone flight session).
- Core annotation format is paired files:
    - image: `.jpg` or `.jpeg`
    - label: `.txt`
    - both files must have the same base name
- Many folders include a `count.txt` summary file.

Folder naming pattern:
`[Date]_[Location-or-Subject]_[DroneModel]_[SensorType]_[OptionalBatchNumber]`

Field notes:
- `Date` uses `YYMMDD` format.
- `SensorType` is expected to include values such as `VIS` (visible spectrum) or `IR` (infrared).

## Preprocessing Rules

### 1. Select source folders

- Iterate through all dataset subfolders.
- Keep only folders whose names contain `VIS`.
- Skip all non-`VIS` folders.

### 2. Match image-label pairs

- Recursively scan each selected folder.
- For every image (`.jpg`/`.jpeg`), find a `.txt` file with the same base name.
- Process only valid pairs. Log or skip unmatched files.

### 3. Generate 640x640 crops

For each valid image-label pair:

- Use label coordinates to locate all actors in the image.
- Create one or more `640x640` crops centered or positioned to include target actors.
- A single source image may produce multiple crops.
- If a crop includes multiple actors, keep all actors that fall inside that crop.
- the actor position inside the new image should be somewhat random and the actor shouldn't be cut in half

### 4. Update labels for each crop

- Convert labels from source-image coordinates to crop-local coordinates.
- Save one updated label file per output crop.

### 5. Build output dataset

- Save each crop and its label as a matched pair in the preprocessed dataset.
- Organize outputs into `train`, `val`, and `test` splits.
- Prevent data leakage by splitting at folder/session level (not random image-level split across the same flight).

## Expected Result

After preprocessing, you should have a clean YOLO-ready dataset with:

- consistent image-label pairing
- correct labels per crop
- leakage-safe `train/val/test` split

### YAML file

```bash
path: /absolute/path/to/your/yolo_dataset 

train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: person
```