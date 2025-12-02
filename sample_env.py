import os
import mujoco as mu
import mujoco.viewer as m
import time
import cv2 as cv
from door_detect import indoor_detect, get_intrinsics, pixel_2_point, get_handle
import numpy as np


def cam_2_world(data, model, point, cam_name: str):
    cam_id = mu.mj_name2id(model, mu.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_rot_matrix = data.cam_xmat[cam_id].reshape(3, 3)
    point_world = cam_pos + np.dot(cam_rot_matrix, point)
    return point_world


def cam_2_world_vec(data, model, vec, cam_name: str):
    cam_id = mu.mj_name2id(model, mu.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_rot_matrix = data.cam_xmat[cam_id].reshape(3, 3)
    point_world = np.dot(cam_rot_matrix, vec)
    return point_world


def get_door_loc(detections, dpth, cx, cy, f):
    door_bbox = detections['door'][0]
    u, v = int(door_bbox[0]), int(door_bbox[1])
    u = max(0, min(u, 639))  # Safety clip
    v = max(0, min(v, 479))

    # Door Location Logic
    dz = dpth[v, u]  # Row, Col
    if 0.1 < dz < 10.0:
        door_loc_robot = pixel_2_point(u, v, dz, cx, cy, f)
        # Vision (+Y down) to MuJoCo (+Y up) conversion
        mu_point = np.array([door_loc_robot[0], -door_loc_robot[1], -door_loc_robot[2]])
        door_loc_world = cam_2_world(data, model, mu_point, 'arm_cam')
        print(f"Door World: {door_loc_world}")
        return door_loc_world


def get_handle_loc(detections, dpth, cx, cy, f):
    # Handle Logic
    handle_bbox = detections['handle'][0]
    _, handle_robot, orint_robot = get_handle(img_bgr, handle_bbox, dpth, cx, cy, f)

    if handle_robot is not None and orint_robot is not None:

        handle_mu = np.array([handle_robot[0], -handle_robot[1], -handle_robot[2]])
        orint_mu = np.array([orint_robot[0],  -orint_robot[1],  -orint_robot[2]])

        # Transform to World
        handle_world = cam_2_world(data, model, handle_mu, 'arm_cam')
        orint_world = cam_2_world_vec(data, model, orint_mu, 'arm_cam')

        print(f"Handle World: {handle_world} | Orientation: {orint_world}")
        return handle_world, orint_world


# === Paths ===
cur_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(cur_path)
env_path = os.path.join(parent_path, 'boston_dynamics_spot', 'scene_arm.xml')

# === Load model ===
try:
    model = mu.MjModel.from_xml_path(env_path)
    data = mu.MjData(model)
    renderer = mu.Renderer(model, height=480, width=640)
    k, cx, cy, f = get_intrinsics(640, 480, 90)
    print("MuJoCo environment loaded successfully.")
except Exception as e:
    print(f"Error loading MuJoCo environment: {e}")
    exit()

# ... (Previous imports and functions rearm_cam the same) ...

# === Launch viewer and camera loop ===
with m.launch_passive(model, data) as viewer:
    print("Viewer started. Use mouse/keyboard to control.")

    while viewer.is_running():
        step_time = time.time()

        mu.mj_step(model, data)
        viewer.sync()

        renderer.update_scene(data, camera="arm_cam")
        img = renderer.render()
        renderer.enable_depth_rendering()
        dpth = renderer.render()
        renderer.disable_depth_rendering()

        # Normalize depth for display
        dpth_norm = cv.normalize(dpth, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        annotated_frame, detections = indoor_detect(img_bgr)

        if len(detections['door']) > 0:
            door_loc_world = get_door_loc(detections, dpth, cx, cy, f)

        if len(detections['handle']) > 0:
            handle_loc_world, handle_orint_world = get_handle_loc(detections, dpth, cx, cy, f)

        dpth_disp = cv.applyColorMap(dpth_norm, cv.COLORMAP_PLASMA)
        cv.imshow("Depth View", dpth_disp)
        cv.imshow("Arm Camera View", annotated_frame)  # Now shows lines if handle detected

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        time_till_next_step = model.opt.timestep - (time.time() - step_time)
        if time_till_next_step > 0:
            time.sleep(time_till_next_step)
