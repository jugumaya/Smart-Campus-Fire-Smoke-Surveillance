import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    """
    Parses CLI configuration tokens for managing custom object inference.
    """
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference with the custom trained multi-class model.")
    parser.add_argument(
        "--source",
        type=str,
        default="smoke-fire-detection-1/train/images",
        help="Path to verification target image file or directory folder structure."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runmodel/detect/train/weights/best.pt",
        help="Path leading to custom trained model parameters."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runmodel/detect/predict",
        help="Target folder directory to store processed visual frames."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence cutoff score to display valid bounding annotations."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    
    # Sequential path checking fallback routine within the new 'runmodel' space
    if not weights_path.exists():
        alternative_paths = [
            Path("runmodel/detect/train2/weights/best.pt"),
            Path("runmodel/detect/train-2/weights/best.pt"),
            Path("best.pt")
        ]
        for path in alternative_paths:
            if path.exists():
                weights_path = path
                break
        else:
            raise FileNotFoundError(f"❌ Core Exception: Could not locate best.pt across expected training directories.")

    # Instantiate the network configuration with active parameters
    model = YOLO(str(weights_path))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading custom weights from: {weights_path}")
    print(f"Running detection on input target: {args.source}")

    # Launch tracking process sequence over designated testing stream
    results = model.predict(
        source=args.source,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        imgsz=640,
        conf=args.conf,
        device="cpu",
    )

    print(f"🎉 Detection routine finished successfully. Structural output saved inside: {save_dir.resolve()}")
    return results

if __name__ == "__main__":
    main()
    
    