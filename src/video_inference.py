from ultralytics import YOLO
from pathlib import Path
import cv2
import sys

model_to_use = "20250623_combined_waste_detection"
model_path = Path(__file__).parent.parent / "models" / model_to_use / "best.pt"
model = YOLO(model_path)

# load video input
video_path = Path(__file__).parent.parent / "tests" / "input_video.mp4"  
print(video_path)
cap = cv2.VideoCapture(str(video_path))
# check if mp4 is opened correctly
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)
# check writing out happens correctly
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    sys.exit(1)
results = model(frame)
annotated_frame = results[0].plot()
frame_height, frame_width = annotated_frame.shape[:2]

# Get FPS (fallback to 25 if not readable)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 25

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
output_path = Path(__file__).parent.parent / "tests" / "output_video_ann.mp4"
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    
    out.write(annotated_frame)  # <-- ✅ This line was missing

    cv2.imshow('YOLO11n Waste Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
