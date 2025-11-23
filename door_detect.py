from ultralytics import YOLO
from pathlib import Path
import os


def door_detect(frame):
    ROOT_PATH = Path(__file__).resolve().parent
    MODEL_PATH = os.path.join(ROOT_PATH, 'models\\best.pt')
    model = YOLO(MODEL_PATH)
    result = model.predict(frame, conf=0.80, verbose=False)[0]
    annotated_frame = result.plot()
    classes = result.boxes.cls.cpu().numpy()
    boxes = result.boxes.xywh.cpu().numpy()
    return annotated_frame, classes, boxes
