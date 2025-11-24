from ultralytics import YOLO
from pathlib import Path
import numpy as np
import os


def door_detect(frame):
    ROOT_PATH = Path(__file__).resolve().parent
    MODEL_PATH = os.path.join(ROOT_PATH, 'models\\best.pt')
    model = YOLO(MODEL_PATH)
    result = model.predict(frame, conf=0.80, verbose=False)[0]
    annotated_frame = result.plot()
    classes = result.boxes.cls.cpu().numpy().round().astype(int)
    boxes = result.boxes.xywh.cpu().numpy().round().astype(int)
    detections = {'handle': [],
                  'door': [],
                  'open_door': []}
    for class_id, box in zip(classes, boxes):
        match class_id:
            case 0:
                detections['door'].append(box)
            case 1:
                detections['handle'].append(box)
            case 2:
                detections['open_door'].append(box)
    return annotated_frame, detections


def get_intrinsics(W, H, fovy):
    cx = W/2
    cy = H/2
    fovy = np.deg2rad(fovy)
    f = H/(2*(np.tan(fovy/2)))
    k = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    return k, cx, cy, f


def pixel_2_point(u, v, dz, cx, cy, f):
    x = (u - cx) * dz / f
    y = (v - cy) * dz / f
    z = dz
    return np.array([x, y, z])
