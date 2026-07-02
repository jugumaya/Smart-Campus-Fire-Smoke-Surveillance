import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO, settings

# =====================================================================
# 1. DATASET INGESTION LAYER
# =====================================================================
print("⏳ Establishing connection to Roboflow Engine via secured token...")
rf = Roboflow(api_key="AS0ghEq3lOYv1SkcCwuy")
project = rf.workspace("smart-campus-fire-smoke-detection").project("smoke-fire-detection-uegm3")
version = project.version(2)
dataset = version.download("yolov8")

print("✅ High-precision multi-class dataset download complete.")

# =====================================================================
# 2. PATH VALIDATION LAYER
# =====================================================================
yaml_path = os.path.join(dataset.location, "data.yaml")

# Open data layout specifications schema for validation
with open(yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

# Hardcode dataset anchor coordinates to patch absolute path requirements
dataset_root_path = dataset.location
yaml_data['path'] = dataset_root_path
yaml_data['train'] = os.path.join("train", "images")

# Dynamic verification step for evaluation validation split directories
valid_images_path = os.path.join(dataset_root_path, "valid", "images")
if os.path.exists(valid_images_path):
    yaml_data['val'] = os.path.join("valid", "images")
else:
    yaml_data['val'] = os.path.join("train", "images")
    print("⚠️ Directory 'valid/images' missing. Reverting target path onto fallback image logs.")

# Dynamic verification step for evaluation testing split directories
if 'test' in yaml_data:
    test_images_path = os.path.join(dataset_root_path, "test", "images")
    if os.path.exists(test_images_path):
        yaml_data['test'] = os.path.join("test", "images")
    else:
        yaml_data['test'] = os.path.join("train", "images")

# Save modifications back to file system to synchronize training paths
with open(yaml_path, 'w') as f:
    yaml.safe_dump(yaml_data, f)

print(f"⚙️ Target configuration patched successfully. Root tracking sync point: {dataset_root_path}")

# =====================================================================
# 3. CRITICAL OPTIMIZATION ROUTINE
# =====================================================================
# Sync global working space configurations to avoid local directory anomalies
settings.update({"datasets_dir": os.getcwd()})

print("🧠 Loading baseline object detection structural weights...")
model = YOLO("yolov8n.pt")

print("🚀 Initializing training loop configuration sequence...")
results = model.train(
    data=yaml_path, 
    epochs=20,                    # Performance learning cycles over dataset
    imgsz=640,                    # Input grid image processing resolution
    device="cpu",                 # Default platform targeting execution core
    project="runmodel/detect",    # Enforced isolation folder directory
    name="train"                  # Target namespace for compiled output weights
)

print("\n🎉 Model optimization sequence complete.")
print("🔑 Base weights preserved safely at location: 'runmodel/detect/train/weights/best.pt'")


