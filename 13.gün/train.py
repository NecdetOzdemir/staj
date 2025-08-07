from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

# Modeli yükle (pre-trained weights ile başlayın)
model = YOLO("yolov8n.pt")  # .pt ile başlayın (daha hızlı öğrenme)

model.train(
    data="/home/necdet/yolo/plaka/sadece_plaka/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    workers=4,
    device=0,
    patience=15,
    optimizer="SGD",
    cos_lr=True,
    lr0=0.01,         # Daha yüksek LR
    cache=True,
    # Augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,        # Plaka için karıştırma gerekmez
    copy_paste=0.0    # Kopyalama gerekmez
)