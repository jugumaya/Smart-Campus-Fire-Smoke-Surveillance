from pathlib import Path
from ultralytics import YOLO

def export_custom_model():
    """
    Compiles PyTorch model parameter files into high-performance graph processing deployment modules.
    """
    print("⏳ Parsing native model weight profiles (.pt)...")
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / "runs" / "detect" / "train" / "weights" / "best.pt",
        base_dir / "runs" / "detect" / "train2" / "weights" / "best.pt",
        base_dir / "runs" / "detect" / "train-2" / "weights" / "best.pt",
        base_dir / "best.pt",
    ]

    weights_path = next((p for p in candidate_paths if p.exists()), None)
    if not weights_path:
        print("❌ Core Exception: No valid configuration models matched deployment specifications targets.")
        return

    print(f"✅ Active graph parameters loaded safely from target: {weights_path}")
    model = YOLO(str(weights_path))

    print("🚀 Exporting active parameters to high-efficiency ONNX production runtime environments...")
    try:
        model.export(format="onnx")
        print("🎉 Graph export operations complete. Production engine active via ONNX.")
    except Exception as exc:
        print("❌ Graph structural processing failed during serialization steps:", exc)

if __name__ == "__main__":
    export_custom_model()
    
    