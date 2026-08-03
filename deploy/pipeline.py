import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. Initialize the SAHI AutoDetectionModel
# SAHI needs to know which backend wrapper to use. 
# Use 'yolov8' for both YOLOv8 and YOLO11, as they share the same Ultralytics API.
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='runs\\sliced_NOMAD_less_bg_actors91-100\\nomad_yolo26n\\weights\\best.pt',       # Use your PyTorch or ONNX file here for laptop testing
    confidence_threshold=0.3,   # Lower this if mannequins are being missed
    device="cpu",               # Change to "cuda:0" when you move to the Jetson
)

# 2. Define the path to a high-res test image (e.g., from your NOMAD or C2A dataset)
image_path = r"C:\Users\Bartek\Desktop\SUAV\wisard\200426_SkookumCreek_Mavic_Mini_VIS_0006\200426_SkookumCreek_Mavic_Mini_VIS_0006_00000001.jpg"

# 3. Perform Sliced Inference
print("Slicing image and running inference...")
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=512,             # Must match the resolution you trained your YOLO model on
    slice_width=512,
    overlap_height_ratio=0.25,    # 25% overlap ensures objects on seams aren't missed
    overlap_width_ratio=0.25,
    perform_standard_pred=False,  # Set to True to also run inference on the full, unsliced image
    postprocess_type="NMS",       # Non-Maximum Suppression merges duplicate boxes
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.5
)

# 4. Extract and Print Coordinates
# This is where you grab the coordinates to calculate your payload drop
for object_prediction in result.object_prediction_list:
    bbox = object_prediction.bbox.to_xyxy() # Returns [xmin, ymin, xmax, ymax]
    score = object_prediction.score.value
    category_name = object_prediction.category.name
    
    # Calculate the centroid (center of the bounding box) for targeting
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    
    print(f"Detected {category_name} (Conf: {score:.2f}) at pixel [{center_x}, {center_y}]")

# 5. Visualize and Save the Results
output_dir = "sahi_results/"
print(f"Exporting visualization to {output_dir}")
result.export_visuals(export_dir=output_dir, file_name="sahi_output")

# (Optional) Display the image immediately using OpenCV
output_image = cv2.imread(f"{output_dir}sahi_output.png")
cv2.imwrite("SAHI Output.png", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
