from ultralytics import YOLO
from pathlib import Path
import numpy as np
import os
import cv2


def indoor_detect(frame):
    ROOT_PATH = Path(__file__).resolve().parent
    MODEL_PATH = os.path.join(ROOT_PATH, 'models/best.pt')
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


def get_handle(src_img, handle_bbox, dpth, cx, cy, f):
    height, width = src_img.shape[:2]

    [x_c, y_c, w_box, h_box] = handle_bbox

    # Calculate corners
    x1 = int(x_c - w_box // 2)
    y1 = int(y_c - h_box // 2)
    x2 = int(x_c + w_box // 2)
    y2 = int(y_c + h_box // 2)

    # SAFETY: Clip coordinates to image bounds to prevent crashes
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    # Crop
    door_crop = src_img[y1:y2, x1:x2]

    # Check if crop is valid (empty check)
    if door_crop.size == 0:
        return src_img, None, None

    # HSV Masking
    door_hsv = cv2.cvtColor(door_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(door_hsv, (0, 0, 97), (15, 97, 210))

    # Place mask on full image
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    # Overlay Visualization
    color_overlay = np.zeros_like(src_img)
    color_overlay[:] = (0, 0, 255)
    color_overlay = cv2.bitwise_and(color_overlay, color_overlay, mask=full_mask)

    output_img = cv2.addWeighted(src_img, 1.0, color_overlay, 0.5, 0)

    v_idxs, u_idxs = np.where(full_mask > 0.0)

    if len(v_idxs) < 10:
        return output_img, None, None

    # Get Depth
    # dpth is accessed as [row, col] -> [v, u]
    dz = dpth[v_idxs, u_idxs]

    valid_mask = (dz > 0.1) & (dz < 3.0)
    if np.sum(valid_mask) < 10:
        return output_img, None, None

    u_valid = u_idxs[valid_mask]
    v_valid = v_idxs[valid_mask]
    z_valid = dz[valid_mask]

    # 3D Math (Now correct because u is X and v is Y)
    x_pts = (u_valid - cx) * z_valid / f
    y_pts = (v_valid - cy) * z_valid / f
    z_pts = z_valid

    points_3d = np.vstack((x_pts, y_pts, z_pts)).T

    # PCA
    centroid = np.mean(points_3d, axis=0)
    centered_data = points_3d - centroid
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Get principal axis
    view_vector = -centroid

    main_axis = eigenvectors[:, -1]
    if np.dot(main_axis, view_vector) < 0:
        main_axis = -main_axis
    main_axis = main_axis / np.linalg.norm(main_axis)

    return output_img, centroid, main_axis


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
