from ultralytics import YOLO
from pathlib import Path
import cv2

model_to_use = "20250530_TACO_to_YOLO_Waste_Detection_with_YOLOv8"

model_path = Path(__file__).parent.parent / "models" / model_to_use / "best.pt"
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Waste Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
