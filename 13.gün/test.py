import cv2
from ultralytics import YOLO
import torch

# CUDA kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Kullanılan cihaz: {device}")

# Eğitilmiş modeli yükle (kendi model yolunu gir)
model = YOLO("/home/necdet/yolo/plaka/sadece_plaka/detect/train4/weights/best.pt")

# Webcam başlat (0 = default kamera)
cap = cv2.VideoCapture(0)

# Kamera çözünürlüğü (isteğe bağlı)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Kamera başlatıldı. 'q' tuşuna basarak çıkabilirsin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile çıkarım yap
    results = model.predict(source=frame, device=0, imgsz=640, conf=0.3, verbose=False)

    # Sonuçları OpenCV formatında çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow("Plaka Tespiti - YOLOv8", annotated_frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamera ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
