import os
import torch
from ultralytics import YOLO

# --- Config ---
DATASET_YAML = r"C:\Users\Bartek\Desktop\IMAV\mission_1\crop_30_50_70\data.yaml"
MODEL_WEIGHTS = "yolo11s.pt"  # or 'yolov8s.pt', 'yolo26s.pt'
PROJECT_DIR = r"./runs/test_try"
RUN_NAME = "yolo11s_crop_30_50_70"

EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 16  # set to float e.g. 0.9 for autobatch if preferred
PATIENCE = 20


def train_yolo_model():
    # Check CUDA availability
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load YOLO model
    print(f"Loading weights: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)

    # Train model
    print(f"Starting training on dataset: {DATASET_YAML}...")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=PATIENCE,
        save=True,
        plots=True,
        exist_ok=True,
    )

    print("\nTraining completed successfully!")
    return results


if __name__ == "__main__":
    train_yolo_model()
