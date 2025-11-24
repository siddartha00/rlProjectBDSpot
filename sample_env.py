import os
import mujoco as mu
import mujoco.viewer as m
import time
import cv2 as cv
from door_detect import door_detect, get_intrinsics, pixel_2_point
import numpy as np

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

# === Launch viewer and camera loop ===
with m.launch_passive(model, data) as viewer:
    print("Viewer started. Use mouse/keyboard to control.")
    print("Renderer type:", type(renderer))
    while viewer.is_running():
        step_time = time.time()

        mu.mj_step(model, data)
        viewer.sync()

        # === Render camera ===

        # Update and render from camera named "arm_cam"
        # Use 'main' Camera for default view

        renderer.update_scene(data, camera="main")  # camera on robot arm
        img = renderer.render()
        renderer.enable_depth_rendering()
        dpth = renderer.render()
        renderer.disable_depth_rendering()
        dpth_norm = cv.normalize(dpth, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)   # convert for OpenCV display
        annotated_frame, detections = door_detect(img_bgr)
        if len(detections['handle']) > 0:
            [u, v] = detections['handle'][0][:2]
            dz = dpth[v, u]
            point = pixel_2_point(u, v, dz, cx, cy, f)
            mu_point = [point[0], -point[1], -point[2]]
            # print(point)
            cam_id = mu.mj_name2id(model, mu.mjtObj.mjOBJ_CAMERA, "main")
            cam_pos = data.cam_xpos[cam_id]
            cam_rot_matrix = data.cam_xmat[cam_id].reshape(3, 3)
            point_world = cam_pos + np.dot(cam_rot_matrix, mu_point)
            print(point_world)
        dpth_disp = cv.applyColorMap(dpth_norm, cv.COLORMAP_PLASMA)
        cv.imshow("Depth View", dpth_disp)
        cv.imshow("Arm Camera View", annotated_frame)

        # Optional: quit camera with 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Match MuJoCo sim time
        time_till_next_step = model.opt.timestep - (time.time() - step_time)
        if time_till_next_step > 0:
            time.sleep(time_till_next_step)
