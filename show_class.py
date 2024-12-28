from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("model/yolov8n.pt")

print(model.names) 