import os
from roboflow import Roboflow

def download_our_dataset():
    """
    Authenticates workspace profiles to fetch target computer vision dataset formats from remote storage points.
    """
    print("⏳ Synchronizing connection with Roboflow Storage Core Engine channels...")
    rf = Roboflow(api_key="AS0ghEq3lOYv1SkcCwuy")

    # Connect to target workspace dataset coordinates
    project = rf.workspace("smart-campus-fire-smoke-detection").project("smoke-fire-detection-uegm3")

    print("🚀 Stream-loading structured bounding annotation coordinates data objects...")
    # Request coordinate matrices matching object tracking specifications layout structures
    dataset = project.version(2).download("yolov8")

    print("🎉 Dataset transfer transaction successfully resolved.")
    print(f"💡 Records saved locally inside active context structure: {dataset.location}")

if __name__ == "__main__":
    download_our_dataset()
    
    