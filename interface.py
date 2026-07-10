import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    """
    Compiles operational input flag options for standalone engine execution parameters.
    """
    parser = argparse.ArgumentParser(description="Run standalone YOLOv8 inference routines.")
    parser.add_argument(
        "--source",
        type=str,
        default="smoke-fire-detection-1/train/images",
        help="Input source filepath configuration matching verification evaluation scopes."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runmodel/detect/train/weights/best.pt",
        help="Path configuration matching specific evaluation weights files."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runmodel/detect/predict",
        help="Enforced output directory target for file storage."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Evaluation precision confidence limit score filters."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    
    # Check alternate storage points if base configurations are missing
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
            raise FileNotFoundError("❌ Core Execution Exception: Missing structural model weights across directory targets.")

    # Instantiate parameters onto execution environment memory
    model = YOLO(str(weights_path))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading custom weights parameters from: {weights_path}")
    print(f"Tracking evaluation verification elements inside: {args.source}")

    # Initialize frame scanning routine operations over standard inputs
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

    print(f"🎉 Detection operations complete. Frame records preserved inside: {save_dir.resolve()}")
    return results

if __name__ == "__main__":
    main()
    
    