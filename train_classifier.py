from ultralytics import YOLO
import os

def train_fire_classifier():
    print("⏳ Initializing YOLOv8 Nano Classification Model...")
    # Load the official pre-trained image classification network
    model = YOLO("yolov8n-cls.pt") 
    
    # 🚀 FIX: Pointing to the exact directory extracted on your computer
    dataset_path = os.path.join(os.getcwd(), "Campus-Fire-Detection-2")
    
    # Double-check if the folder actually exists to avoid runtime crashes
    if not os.path.exists(dataset_path):
        print(f"❌ Error: The directory {dataset_path} does not exist!")
        print("💡 Please make sure 'Campus-Fire-Detection-2' is in your current working directory.")
        return

    print(f"✅ Dataset path verified: {dataset_path}")
    print("🚀 Starting Classification Training Loop...")
    
    # Train the classifier for 20 epochs across your local dataset images
    model.train(
        data=dataset_path,
        epochs=20,
        imgsz=640,
        device="cpu"  # Optimizing for your 12th Gen Intel i3 CPU pipeline
    )
    
    print("🎉 Classification model training complete!")
    print("🔑 Weights saved inside: runs/classify/train/weights/best.pt")

if __name__ == "__main__":
    train_fire_classifier()
    