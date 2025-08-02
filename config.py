import os

# Model Configuration
MODEL_PATH = "runs/segment/train/weights/best.pt"  # Update with your model path

# Dataset Paths
VAL_DATASET_PATH = "datasets/val/images"    # Update with your validation images path
TEST_DATASET_PATH = "datasets/test/images"  # Update with your test images path
GT_PARQUET_PATH = "ground_truth.parquet"    # Update with your ground truth file path

# Output Configuration
OUTPUT_DIR = "/home/praneeth/Desktop/eye_pop_assignment/ep-sai-praneeth-potladurthy/outputs"

# Processing Parameters
IOU_THRESHOLD = 0.5  # IoU threshold for matching predictions to ground truth
CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold for predictions
POLYGON_APPROXIMATION_EPSILON = 0.02  # Epsilon for polygon approximation

# Class Names (update based on your dataset)
CLASS_NAMES = ["roof"]  # Add your class names here
