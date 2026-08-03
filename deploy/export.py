from ultralytics import YOLO

BEST_WEIGHTS_PATH = r"runs\sliced_NOMAD_less_bg_actors91-100\nomad_yolo26n\weights\best.pt"
model = YOLO(BEST_WEIGHTS_PATH)

model.export(
    format="engine",
    quantize=16,       # Enables FP16 precision
    dynamic=True,    # Allows dynamic batching for SAHI slices
    workspace=4,      # Allocates 4GB of RAM for the conversion process
    device=0,
)

# ---------------------------------------------------------
# OPTIONAL: If you must use INT8 for memory savings
# ---------------------------------------------------------
# model.export(
#     format="engine",
#     int8=True,
#     data="dataset.yaml",  # Required: paths to your calibration images
#     dynamic=True,
#     workspace=4
# )