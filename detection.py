import torch
import numpy as np
from ultralytics import YOLO
from config import DEVICE, YOLO_MODEL_PATH

def load_model():
    """Loads the YOLO model."""
    model = YOLO(YOLO_MODEL_PATH).to(DEVICE)
    return model

def detect_plates(model, frame):
    """Runs YOLO inference to detect license plates in a frame."""
    with torch.no_grad():
        results = model(frame, imgsz=1280, device=DEVICE,verbose=False)
    
    detections = [
        [*map(int, box), conf.item()]
        for result in results
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf)
        if conf.item() > 0.5
    ]

    return np.array(detections) if detections else np.empty((0, 5))

