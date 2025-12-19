import os
import mujoco as mu
import mujoco.viewer as m
import time
import cv2 as cv
import numpy as np

print(mu.__version__)

# === Paths ===
cur_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(cur_path)
env_path = os.path.join(parent_path, 'boston_dynamics_spot', 'scene_arm.xml')

# === Load model ===
try:
    model = mu.MjModel.from_xml_path(env_path)
    data = mu.MjData(model)
    renderer = mu.Renderer(model)
    print("MuJoCo environment loaded successfully.")
except Exception as e:
    print(f"Error loading MuJoCo environment: {e}")
    exit()

# def get_feet_contact(model: mu.MjModel, data: mu.MjData, foot_geom_names, ground_geom_name="groundplane"):
#     """
#     Returns an array of 1/0 indicating whether each foot is in contact with the ground.
    
#     Args:
#         model: MuJoCo model
#         data: MuJoCo data
#         foot_geom_names: list of strings, names of the foot geoms
#         ground_geom_name: string, name of the ground geom
#     Returns:
#         np.array of shape (num_feet,), 1 if in contact, 0 if in air
#     """
#     feet_geom_ids = [model.geom(name).id for name in foot_geom_names]
#     ground_geom_id = model.geom(ground_geom_name).id
    
#     # initialize contact array
#     feet_contact = np.zeros(len(feet_geom_ids), dtype=int)
    
#     # loop over all contacts in this step
#     for i in range(data.ncon):
#         contact = data.contact[i]
#         g1, g2 = contact.geom1, contact.geom2
#         for j, foot_id in enumerate(feet_geom_ids):
#             if (g1 == foot_id and g2 == ground_geom_id) or (g2 == foot_id and g1 == ground_geom_id):
#                 feet_contact[j] = 1  # foot is in contact
    
#     return feet_contact

def get_geom_id(model: mu.MjModel, name: str):
    for i in range(model.ngeom):
        geom_name = model.geom(i).name  # this is already a str
        if geom_name == name:           # exact match
            return i
    raise ValueError(f"Geom name {name} not found")

def get_feet_contact(model: mu.MjModel, data: mu.MjData, foot_names):
    foot_geom_ids = [get_geom_id(model, name) for name in foot_names]
    contacts = np.zeros(len(foot_names), dtype=int)

    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in foot_geom_ids:
            contacts[foot_geom_ids.index(c.geom1)] = 1
        if c.geom2 in foot_geom_ids:
            contacts[foot_geom_ids.index(c.geom2)] = 1

    return contacts

# Example
# foot_names = ["FL", "FR", "HL", "HR"]
# contacts = get_feet_contact(model, data, foot_names)
# print("Feet in contact:", contacts)


# === Example usage ===

import numpy as np

def get_foot_positions(model, data):
    """
    Returns world positions of all four feet.
    """
    foot_names = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
    foot_positions = {}

    for name in foot_names:
        body_id = model.body(name).id
        # data.xpos[body_id] gives [x, y, z] of body in world frame
        foot_positions[name] = np.array(data.xpos[body_id])

    return foot_positions




# === Launch viewer and camera loop ===
with m.launch_passive(model, data) as viewer:
    print("Viewer started. Use mouse/keyboard to control.")
    print("Renderer type:", type(renderer))
    print(model.opt.timestep)
    while viewer.is_running():
        # mu.mj_resetData(model,data)
        # mu.mj_forward(model,data)
        # mu.mj_resetData(model, data)
        step_time = time.time()

        # print(data.qpos)
        # print(data.ctrl)
        # print(data.contact)
        # foot_names = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]
        default_pos = [0,0.8,-1.5,0,0.8,-1.5,0,1.0,-1.5,0,1.0,-1.5]
        # data.qpos[0:2] = np.array([
        #         np.random.uniform(5.0, 20.0),   # x offset
        #         np.random.uniform(-7.5, -3.5),   # y offset
        #         # 0.35                            # z height above ground
        # ])
        data.ctrl[:12] = default_pos
        foot_names = ["FL", "FR", "HL", "HR"]
        contacts = get_feet_contact(model, data, foot_names)
        body_id = model.body("fr_lleg").id
        foot_vel = data.cvel[body_id][3]  # linear velocity [vx, vy, vz]

        total_joint_torque = (
            data.qfrc_actuator +
            data.qfrc_bias +
            data.qfrc_constraint +
            data.qfrc_applied
        )

        # print(data.actuator_force)
        print(get_foot_positions(model,data))
        # print(data.qpos[2])

        # print("Feet contact array:", contacts)


        # Simulate one step
        # for i in range(model.njnt):
        #     jname = model.joint(i).name
        #     qpos_addr = model.jnt_qposadr[i]
        #     qvel_addr = model.jnt_dofadr[i]

        #     # Some joints (like free joints) use 7 qpos (quat + pos)
        #     nq = 7 if model.jnt_type[i] == mu.mjtJoint.mjJNT_FREE else 1
        #     nv = 6 if model.jnt_type[i] == mu.mjtJoint.mjJNT_FREE else 1

        #     qpos_vals = data.qpos[qpos_addr:qpos_addr + nq]
        #     qvel_vals = data.qvel[qvel_addr:qvel_addr + nv]

        #     print(f"Joint: {jname}")
        #     print(f"  Type: {model.jnt_type[i]}")
        #     print(f"  qpos idx [{qpos_addr}:{qpos_addr + nq}] → {qpos_vals}")
        #     print(f"  qvel idx [{qvel_addr}:{qvel_addr + nv}] → {qvel_vals}")
        mu.mj_step(model, data)
        viewer.sync()

        # === Render camera ===

        # Update and render from camera named "arm_cam"
        # Use 'main' Camera for default view

        # renderer.update_scene(data, camera="arm_cam")  # camera on robot arm
        # img = renderer.render()
        # renderer.enable_depth_rendering()
        # dpth = renderer.render()
        # renderer.disable_depth_rendering()
        # dpth_norm = cv.normalize(dpth, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        # img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)   # convert for OpenCV display
        # dpth_disp = cv.applyColorMap(dpth_norm, cv.COLORMAP_JET)
        # cv.imshow("Depth View", dpth_disp)
        # cv.imshow("Arm Camera View", img_bgr)

        # Optional: quit camera with 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Match MuJoCo sim time
        time_till_next_step = model.opt.timestep - (time.time() - step_time)
        if time_till_next_step > 0:
            time.sleep(time_till_next_step)
