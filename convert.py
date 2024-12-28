from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolov8n.pt")

model.export(format="engine",half = True, simplify=True, imgsz=640)

