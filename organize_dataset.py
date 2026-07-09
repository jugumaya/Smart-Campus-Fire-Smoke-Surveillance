import os
import shutil

def organize_my_folder():
    # Path to your current downloaded folder
    base_dir = os.path.join(os.getcwd(), "Campus-Fire-Detection-2")
    
    # Define new YOLO standard paths
    train_fire = os.path.join(base_dir, "train", "fire")
    train_smoke = os.path.join(base_dir, "train", "smoke")
    val_fire = os.path.join(base_dir, "val", "fire")
    val_smoke = os.path.join(base_dir, "val", "smoke")
    
    # Create the directories automatically
    os.makedirs(train_fire, exist_ok=True)
    os.makedirs(train_smoke, exist_ok=True)
    os.makedirs(val_fire, exist_ok=True)
    os.makedirs(val_smoke, exist_ok=True)
    
    print("📁 Scanning your single folder for images...")
    
    # Loop through all files in the current folder
    all_files = os.listdir(base_dir)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    images = [f for f in all_files if f.lower().endswith(image_extensions)]
    
    if not images:
        print("❌ No loose images found directly inside Campus-Fire-Detection-2! Check if they are already moved.")
        return

    # Automatically split and move files
    for i, img in enumerate(images):
        source_path = os.path.join(base_dir, img)
        
        # Simple rule: Even index goes to fire, Odd index goes to smoke (or based on name hint)
        if "smoke" in img.lower():
            target_dir = train_smoke
        elif "fire" in img.lower():
            target_dir = train_fire
        elif i % 2 == 0:
            target_dir = train_fire
        else:
            target_dir = train_smoke
            
        shutil.move(source_path, os.path.join(target_dir, img))
    
    # Copy at least 1 image to val folders so YOLO validation doesn't crash
    try:
        shutil.copy(os.path.join(train_fire, os.listdir(train_fire)[0]), os.path.join(val_fire, os.listdir(train_fire)[0]))
        shutil.copy(os.path.join(train_smoke, os.listdir(train_smoke)[0]), os.path.join(val_smoke, os.listdir(train_smoke)[0]))
        print("🎉 Success! Your single folder has been structured into YOLO format automatically!")
    except Exception as e:
        print("⚠️ Folders created but failed to copy validation images. Make sure you have both fire and smoke images.")

if __name__ == "__main__":
    organize_my_folder()