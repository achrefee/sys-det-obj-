from ultralytics import YOLO
import os

def main():
    # 1. Load YOLOv8 model (nano version)
    model = YOLO("yolov8m.pt")  # <-- changement ici

    # 2. Train the model
    checkpoint_path = os.path.join("runs", "detect", "BDD100K_Training", "run1", "weights", "last.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        results = model.train(
            data="data.yaml",
            resume=True
        )
    else:
        print("Starting fresh training...")
        results = model.train(
            data="data/data.yaml", 
            epochs=3,
            imgsz=640, 
            device=0, 
            batch=8,
            workers=2,
            project="yolov8m",
            name="run1",
            exist_ok=True 
        )

if __name__ == "__main__":
    main()
