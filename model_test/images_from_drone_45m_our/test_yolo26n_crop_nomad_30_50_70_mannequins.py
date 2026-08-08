import os
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- Configuration ---
PREDICT_IMAGES_DATASET_PATH = r"C:\Users\Bartek\Desktop\SUAV\SP_test_45m_people_mannequin\images"
OUTPUT_PATH = r"model_test\images_from_drone_45m_our\results_v2"
MODEL_WEIGHTS_PATH = r"C:\Users\Bartek\Downloads\best.pt"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Model Initialization ---
# Assuming you are using an Ultralytics YOLOv8/YOLO11 model. 
# If it's YOLOv5, change model_type to "yolov5".
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", 
    model_path=MODEL_WEIGHTS_PATH,
    confidence_threshold=0.6, # Adjust based on your model's strictness
    device="cuda:0"           # Change to "cpu" if you are not using a GPU
)

# --- Processing Loop ---
# Grab all common image types from the input folder
supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
image_paths = [p for p in Path(PREDICT_IMAGES_DATASET_PATH).iterdir() if p.suffix.lower() in supported_extensions]

print(f"Found {len(image_paths)} images to process.")

for img_path in image_paths:
    print(f"Processing: {img_path.name}...")
    
    # Perform the sliced prediction
    result = get_sliced_prediction(
        img_path.as_posix(),
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2, # 20% vertical overlap (102 pixels)
        overlap_width_ratio=0.2,  # 20% horizontal overlap (102 pixels)
        perform_standard_pred=False # Set to True if you also want to run a prediction on the full un-sliced image
    )
    
    # Save the resulting image with bounding boxes drawn
    # SAHI automatically saves this as an image file (e.g., .png) 
    result.export_visuals(
        export_dir=OUTPUT_PATH,
        file_name=img_path.stem, 
        text_size=0.5,           # Adjust label text size
        rect_th=2                # Adjust bounding box thickness
    )

print(f"Inference complete! Saved {len(image_paths)} visualized images to '{OUTPUT_PATH}'.")