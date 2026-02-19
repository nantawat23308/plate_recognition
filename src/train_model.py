from ultralytics import YOLO
from roboflow import Roboflow
import os
from dotenv import load_dotenv
from src.annotate import extract_corners
load_dotenv()


def prepare_data():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW"))
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    version = project.version(13)
    dataset = version.download("yolo26")
    return dataset

def main():
    model = YOLO("../yolo26n.pt")
    dataset = prepare_data()
    model.train(
            data=f"{dataset.location}/data.yaml",
            epochs=1,
            imgsz=640,
            batch=16,
            device=0,
            name="yolo26n-plate",

            # --- OPTIMIZER SETTINGS ---
            optimizer='AdamW',
            lr0=0.001,  # Initial learning rate # Good starting rate for AdamW
            lrf=0.01,   # Final learning rate (multiplier)
            cos_lr=True,  # Cosine scheduler helps convergence at the end
            momentum=0.937,  # Momentum for SGD
            weight_decay=0.0005,  # L2 regularization
            warmup_epochs=3.0,  # Warmup epochs (fractions ok)
            warmup_momentum=0.8,  # Warmup initial momentum
            warmup_bias_lr=0.1,  # Warmup initial bias lr

            # --- LOSS / HYPERPARAMETERS ---
            box=10.0,  # Box loss gain # Focus MORE on box accuracy (critical for OCR cropping)
            cls=0.5,  # Class loss gain # Focus LESS on classification (since it's just 1 class)
            dfl=1.5,  # DFL loss gain # Distribution Focal Loss (fine-tuning box precision)
            label_smoothing=0.0,  # Label smoothing (fraction)
            nbs=64,  # Nominal batch size

            # --- AUGMENTATION (Avoid bad data) ---
            flipud=0.0,  # Vertical flip augmentation probability # DISABLE vertical flip (cars don't fly)
            fliplr=0.5,  # Horizontal flip augmentation probability # Horizontal flip is OK (mirror image)
            mosaic=1.0,  # Mosaic augmentation probability # Keep Mosaic on (helps small object detection)
            mixup=0.0,  # Mixup augmentation probability
            degrees=5.0,  # Rotation degrees # Only slight rotation (+/- 10 deg)

            # --- PERFORMANCE ---
            rect=True,  # Good for non-square images (optimizes batch shape)
            patience=10  # Stop early if no improvement for 10 epochs
        )
    print("Training Complete!")
