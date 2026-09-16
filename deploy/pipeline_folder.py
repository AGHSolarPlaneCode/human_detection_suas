import cv2
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. Initialize the SAHI AutoDetectionModel
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=r"C:\Users\Bartek\Downloads\astra_imav.pt",
    confidence_threshold=0.35,
    device="cpu", # Note: "cpu" will be quite slow for a whole folder
)

# 2. Define input path (can be a file OR a folder) and output directory
input_path = Path(r"C:\Users\Bartek\Desktop\IMAV\pojazdy") 
output_dir = Path(r"C:\Users\Bartek\Desktop\SUAV\human_detection_suas\sahi_results_34_imav\final")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Determine if input is a single file or a directory containing images
if input_path.is_file():
    image_paths = [input_path]
elif input_path.is_dir():
    image_paths = []
    # Search for common image formats
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(input_path.rglob(ext))
else:
    raise ValueError("The provided path does not exist.")

print(f"Found {len(image_paths)} image(s). Starting inference...")

# 3. Loop through all found images
for img_path in image_paths:
    print(f"\n--- Slicing and inferring: {img_path.name} ---")
    
    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5
    )

    # 4. Extract and Print Coordinates
    for object_prediction in result.object_prediction_list:
        bbox = object_prediction.bbox.to_xyxy() 
        score = object_prediction.score.value
        category_name = object_prediction.category.name
        
        # Calculate the centroid
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        print(f"Detected {category_name} (Conf: {score:.2f}) at pixel [{center_x}, {center_y}]")

    # 5. Visualize and Save the Results
    # Append the original filename to the output so files aren't overwritten
    output_filename = f"sahi_{img_path.stem}"
    result.export_visuals(export_dir=str(output_dir), file_name=output_filename)

print(f"\nBatch processing complete. All visualizations saved to {output_dir}")