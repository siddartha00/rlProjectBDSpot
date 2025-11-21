import os
import mujoco as mu
import mujoco.viewer as m
import time
import cv2 as cv

# === Paths ===
cur_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(cur_path)
env_path = os.path.join(parent_path, 'boston_dynamics_spot', 'scene_arm.xml')

# === Load model ===
try:
    model = mu.MjModel.from_xml_path(env_path)
    data = mu.MjData(model)
    renderer = mu.Renderer(model, height=480, width=640)
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

        renderer.update_scene(data, camera="arm_cam")  # camera on robot arm
        img = renderer.render()
        renderer.enable_depth_rendering()
        dpth = renderer.render()
        renderer.disable_depth_rendering()
        dpth_norm = cv.normalize(dpth, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)   # convert for OpenCV display
        dpth_disp = cv.applyColorMap(dpth_norm, cv.COLORMAP_JET)
        cv.imshow("Depth View", dpth_disp)
        cv.imshow("Arm Camera View", img_bgr)

        # Optional: quit camera with 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Match MuJoCo sim time
        time_till_next_step = model.opt.timestep - (time.time() - step_time)
        if time_till_next_step > 0:
            time.sleep(time_till_next_step)
