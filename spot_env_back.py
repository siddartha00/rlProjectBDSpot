# import torch
import math
import os
import mujoco as mu
import mujoco.viewer as m
import time
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
# === Paths ===
cur_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(cur_path)
env_path = os.path.join(parent_path, 'boston_dynamics_spot', 'scene_arm.xml')


default_joint_angles = {
    "fl_hx": 0.0,
    "fl_hy": 0.8,
    "fl_kn": -1.5,
    "fr_hx": 0.0,
    "fr_hy": 0.8,
    "fr_kn": -1.5,
    "hl_hx": 0.0,
    "hl_hy": 1.0,
    "hl_kn": -1.5,
    "hr_hx": 0.0,
    "hr_hy": 1.0,
    "hr_kn": -1.5,
}

arm_folded = np.array([
    0.0,    # shoulder yaw
    -3.3,    # shoulder pitch up
    3.3,   # elbow folded inward
    -1.4,    # wrist pitch up
    0.0     # wrist yaw
])


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

    return np.array(contacts)

def get_foot_velocities(model, data):
    """
    Returns linear velocities (vx, vy, vz) for all four feet in world frame.
    Works for Unitree-style models with bodies named:
    ['fl_lleg', 'fr_lleg', 'hl_lleg', 'hr_lleg'].

    Returns:
        dict: {
            "fl": np.array([vx, vy, vz]),
            "fr": np.array([vx, vy, vz]),
            "hl": np.array([vx, vy, vz]),
            "hr": np.array([vx, vy, vz])
        }
    """
    foot_names = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
    foot_vels = []

    for name in foot_names:
        body_id = model.body(name).id
        # cvel: [angular(3), linear(3)]
        # We only take the linear velocity part in world frame.
        foot_vels.append(np.array(data.cvel[body_id][3]))

    return np.array(foot_vels)

def get_foot_positions(model, data):
    """
    Returns world positions of all four feet.
    """
    foot_names = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
    foot_positions = []

    for name in foot_names:
        body_id = model.body(name).id
        foot_positions.append(data.xpos[body_id][2] - 0.2)

    return np.array(foot_positions)



class SpotEnv:
    def __init__(self, num_obs = 52, num_actions = 12, num_commands = 4, show_viewer=True, device="cuda", num_steps_per_ep = 2000):
        # self.device = torch.device(device)
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_commands = num_commands    # [goal_x,goal_y]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.002  # control frequency on real robot is 50hz
        self.observations = []
        self.current_cmd = [0.5,0,0]
        # self.prev_action = np.asarray([0,0.8,-0.5,0,0.8,-0.5,0,1.0,-1.5,0,1.0,-1.5])
        self.prev_action = np.asarray([0,0.0,0.0,0,0.0,0.0,0,0.0,0.0,0,0.0,0])
        self.default_pos = [0,0.8,-1.5,0,0.8,-1.5,0,1.0,-1.5,0,1.0,-1.5]
        # self.default_pos = [0.005,-0.04,-0.2846,0.0053,-0.0443,-0.286,-0.00534,-0.0297,-0.272,-0.0055,-0.0297,-0.273]
        self.num_steps_per_ep = num_steps_per_ep
        self.current_step = 0
        self.show_viewer = show_viewer
        self.feet_air_time = np.array([0,0,0,0],dtype=float)
        self.terminate_1 = False
        self.base_height = 0.517

        # === Load model ===
        try:
            self.model = mu.MjModel.from_xml_path(env_path)
            self.data = mu.MjData(self.model)
            renderer = mu.Renderer(self.model)
            print("MuJoCo environment loaded successfully.")
        except Exception as e:
            print(f"Error loading MuJoCo environment: {e}")
            exit()

        self.foot_vels = get_foot_velocities(self.model,self.data)

        self.start_pos = self.data.qpos[0:3].copy()  # x, y, z
        self.target_distance = 2
        self.foot_names = ["FL", "FR", "HL", "HR"]
        self.feet_contacts = get_feet_contact(self.model, self.data, self.foot_names)
        self.feet_positions = get_foot_positions(self.model,self.data)
        if self.show_viewer:
            self.viewer = mu.viewer.launch_passive(self.model, self.data)


    def reset(self):       # reset bot to a default position and set joints to default angles
        mu.mj_resetData(self.model, self.data)
        joint_angles = list(default_joint_angles.values())
        # print(joint_angles)
        self.data.qpos[0:2] = np.array([
                np.random.uniform(5.0, 20.0),   # x offset
                np.random.uniform(-7.5, -3.5),   # y offset
                # 0.35                            # z height above ground
            ])
        yaw = np.random.uniform(-1, 1)
        quat = self._yaw_to_quat(yaw)
        self.data.qpos[3:7] = quat
        self.data.ctrl[:12] = joint_angles
        # self.data.qpos[7:19] = joint_angles
        self.data.qvel[:] = 0
        # mu.mj_forward(self.model, self.data)
        mu.mj_step(self.model, self.data)

        self.start_pos = self.data.qpos[0:3].copy()  # x, y, z
        self.sample_velocity_command()
        
        def_pos = list(default_joint_angles.values())
        new_pos = []
        for i in range(len(def_pos)):
            new_pos.append(def_pos[i])

        # self.data.ctrl[:12] = 
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # convert to (x, y, z, w)

        # ---- Base velocities ----
        base_lin_vel_world = self.data.qvel[0:3]
        base_ang_vel_body = self.data.qvel[3:6]  # already in body frame in MuJoCo

        # Transform linear velocity to body frame
        base_lin_vel_body = rot.apply(base_lin_vel_world, inverse=True)
        current_pos = self.data.qpos[0:3]

        distance_traveled = np.linalg.norm(current_pos[:2] - self.start_pos[:2])  # XY-plane
        remaining_distance = self.target_distance - distance_traveled

        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
        q = self.data.qpos[7:19]      # joint angles
        qdot = self.data.qvel[6:18]   # joint velocities
        default_pos = list(default_joint_angles.values())
        final_pos = []
        for i in range(len(q)):
            final_pos.append(q[i]-default_pos[i])

        # ---- Commands (given externally, e.g., sampled target [vx, vy, yaw_rate]) ----
        cmd = np.asarray(self.current_cmd)  # shape (3,)

        self.prev_action = np.asarray([0,0.0,0.0,0,0.0,0.0,0,0.0,0.0,0,0.0,0])
        torques_applied = self.data.actuator_force


        g_world = np.array([0, 0, -9.8])
        g_body = rot.apply(g_world, inverse=True)

        # ---- Previous action ----

        # ---- Combine all ----
        self.observations = [
            np.array(base_lin_vel_body) * 2,     # (3)
            np.array(base_ang_vel_body) * 0.25,     # (3)
            roll,
            pitch,
            [cmd[0]*2,cmd[1]*2,cmd[2]*0.25],                   # (3)
            final_pos,                     # (12)
            np.array(qdot) * 0.05,                  # (12)
            self.prev_action,            # (12)
            remaining_distance * 0.5,
            self.data.qpos[2],
            # torques_applied[:12],
            g_body
        ]

        # print(g_body)

        obs_flatten = np.concatenate([
            np.ravel(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
            for x in self.observations
        ])

        start_time = self.data.time
        timeout_ = 0 

        while True:
            self.feet_contacts = get_feet_contact(self.model, self.data, self.foot_names)
            check = np.array([1,1,1,1])
            # print(self.feet_contacts)
            if np.array_equal(self.feet_contacts,check) :
            # and abs(base_lin_vel_body[2]/2) < 0.05:
                break
            mu.mj_step(self.model, self.data)

            if self.data.time - start_time > 2.0:
                print("Timeout: robot did not settle within 2 seconds")
                timeout_ = 1
                break
        

        # time.sleep(2.0)

        self.feet_contacts = get_feet_contact(self.model, self.data, self.foot_names)
        self.terminate_1  = False
        self.foot_vels = get_foot_velocities(self.model,self.data)
        self.feet_positions = get_foot_positions(self.model,self.data)

        return obs_flatten, timeout_

    def _yaw_to_quat(self, yaw):
        """Convert yaw angle (radians) to quaternion [w, x, y, z]"""
        qw = np.cos(yaw / 2)
        qx, qy, qz = 0, 0, np.sin(yaw / 2)
        return np.array([qw, qx, qy, qz])
    
    def sample_velocity_command(self, max_forward_speed=2.0, max_yaw_rate=1.0):

        vx = np.random.uniform(0, max_forward_speed)  # forward speed
        vy = 0.0                                        # no lateral movement
        wz = 0.0                                        # no rotation

        self.target_distance = np.random.uniform(0.0, 3.0)
        # print(vx)

        self.current_cmd = np.array([-0.1, vy, wz])
        return self.current_cmd

    def action_dims(self):
        return self.num_actions
    

    def obs_dims(self):
        return self.num_obs




    def step(self,actions_motor_pos,view=False):        # step the environment by providing the control torques
        
        def_pos = list(default_joint_angles.values())
        new_pos = []
        for i in range(len(def_pos)):
            if actions_motor_pos[i] > 100:
                actions_motor_pos[i] = 100
            elif actions_motor_pos[i] < -100:
                actions_motor_pos[i] = -100
            new_pos.append(0.4*actions_motor_pos[i] + self.default_pos[i])

        # self.data.ctrl[:12] = 
        self.data.ctrl[:12] = new_pos
        self.data.qpos[19:24] = arm_folded.copy()
        # self.data.ctrl[12:17] = arm_folded.copy()
        mu.mj_step(self.model, self.data)

        if self.show_viewer:
            if self.viewer.is_running():
                self.viewer.sync()
            # Optional: if viewer is closed by user, handle it
            else:
                print("Viewer closed by user.")
                self.viewer = None
        self.foot_vels = get_foot_velocities(self.model,self.data)
        self.feet_contacts = get_feet_contact(self.model, self.data, self.foot_names)
        self.feet_positions = get_foot_positions(self.model,self.data)
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # convert to (x, y, z, w)

        # ---- Base velocities ----
        base_lin_vel_world = self.data.qvel[0:3]
        base_ang_vel_body = self.data.qvel[3:6]  # already in body frame in MuJoCo

        # Transform linear velocity to body frame
        base_lin_vel_body = rot.apply(base_lin_vel_world, inverse=True)
        current_pos = self.data.qpos[0:3]

        distance_traveled = np.linalg.norm(current_pos[:2] - self.start_pos[:2])  # XY-plane
        remaining_distance = self.target_distance - distance_traveled

        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)


        # # ---- Gravity projection in body frame ----
        g_world = np.array([0, 0, -9.8])
        g_body = rot.apply(g_world, inverse=True)

        # ---- Joint states ----
        q = self.data.qpos[7:19]      # joint angles
        qdot = self.data.qvel[6:18]   # joint velocities
        default_pos = list(default_joint_angles.values())
        final_pos = []
        for i in range(len(q)):
            final_pos.append(q[i]-default_pos[i])

        # ---- Commands (given externally, e.g., sampled target [vx, vy, yaw_rate]) ----
        cmd = np.asarray(self.current_cmd)  # shape (3,)

        # ---- Combine all ----
        self.observations = [
            np.array(base_lin_vel_body) * 2,     # (3)
            np.array(base_ang_vel_body) * 0.25,     # (3)
            roll,
            pitch,
            [cmd[0]*2,cmd[1]*2,cmd[2]*0.25],                   # (3)
            final_pos,                     # (12)
            np.array(qdot) * 0.05,                  # (12)
            self.prev_action,            # (12)
            remaining_distance * 0.5,
            self.data.qpos[2],
            # torques_applied[:12],
            g_body
        ]

        # obs_flatten = np.concatenate(self.observations).flatten()
        obs_flatten = np.concatenate([
            np.ravel(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
            for x in self.observations
        ])





        self.prev_action = np.asarray(actions_motor_pos)

        rews = self.rewards(actions_motor_pos)
        done = self.terminate()
        
        self.current_step+=1

        if self.terminate_1 == True:
            rews-=100        

        return obs_flatten, rews, done, None

    # def position_to_torquePD(self,joint_motor_positions_diff):   # convert joint positions to respective torques using PDs
    #     # joint_names = list(default_joint_angles.keys())
    #     # def_joint_angles = [default_joint_angles[name] for name in joint_names]
    #     def_joint_angles = list(default_joint_angles.values())
    #     sig_a = 0.2
    #     required_motor_positions = []
    #     current_motor_positions = self.data.qpos[7:19]
    #     current_motor_vels = self.data.qvel[6:18]
    #     for i in range(len(def_joint_angles)):
    #         required_motor_positions.append(def_joint_angles[i] + sig_a*joint_motor_positions_diff[i])
    #     error_positions = []
    #     for i in range(len(def_joint_angles)):
    #         error_positions.append(required_motor_positions[i] - current_motor_positions[i])
        
    #     kp = 50.0
    #     kd = 1.0

    #     final_torques = []

    #     torque_limits = [-108.79,97.0]

    #     for i in range(len(def_joint_angles)):
    #         joint_torque = (kp*error_positions[i])-(kd*current_motor_vels[i])
    #         if joint_torque < torque_limits[0]:
    #             joint_torque = torque_limits[0]
    #         if joint_torque > torque_limits[1]:
    #             joint_torque = torque_limits[1]
    #         final_torques.append(joint_torque)

    #     return final_torques


    def get_observation(self):        # return the observation vector
        return self.observations
    
    # linear tracking
    def _reward_tracking_lin_vel(self):
        current_lin_x = self.observations[0][0]
        current_lin_y = self.observations[0][1]
        lin_rew = ((self.observations[4][0] - current_lin_x)*(self.observations[4][0] - current_lin_x)) + ((self.observations[4][1] - current_lin_y)*(self.observations[4][1] - current_lin_y))
        tracking_sigma = 0.25
        lin_rew = np.exp(-lin_rew/tracking_sigma)
        return lin_rew
    
    def _reward_tracking_lin_vel_directional(self):
        current_lin_x = self.observations[0][0] / 2.0  # Undo scaling
        desired_lin_x = self.current_cmd[0]
        
        # Direction match reward (strong)
        if desired_lin_x * current_lin_x > 0:  # Same direction
            direction_reward = 0.5
        else:
            direction_reward = 0.0
        
        # Magnitude match reward
        vel_error = abs(desired_lin_x - current_lin_x)
        magnitude_reward = np.exp(-vel_error / 0.2)
        
        return direction_reward + magnitude_reward
    
    # angular tracking
    def _reward_tracking_ang_vel(self):
        current_ang_z = self.observations[1][2]
        lin_rew = ((self.observations[4][2] - current_ang_z)*(self.observations[4][2] - current_ang_z))
        tracking_sigma = 0.25
        lin_rew = np.exp(-lin_rew/tracking_sigma)
        return lin_rew
    
    # def _reward_lin_vel_z(self):
    #     return self.observations[0][2]*self.observations[0][2]
    

    # penalizing rapid changes in actions
    def _reward_action_rate(self, current_action):
        rew = 0
        for i in range(len(current_action)):
            rew+= (current_action[i]-self.prev_action[i])*(current_action[i]-self.prev_action[i])
        
        return rew
    

    # For trying to maintain stability at 0 velocity (not applied when movement is required)
    
    def _reward_similar_to_default(self):
        rew = 0
        for i in range(len(self.observations[5])):
            rew += (self.observations[5][i] - self.default_pos[i])*(self.observations[5][i] - self.default_pos[i])
        
        if self.observations[4][0]/2 < 0.1 and self.observations[0][0]/2 < 0.1:
            return rew
        else:
            return 0
    
    # penalizing changes in pitch and yaw
    def _reward_roll_pitch_(self):
        roll, pitch = self.observations[2]*180/3.14, self.observations[3]*180/3.14
        reward = (roll*roll) + (pitch*pitch)
        return reward
    
    # penalizing velocities in pitch and roll
    def _reward_ang_vel_xy(self):
        rew = (self.observations[1][0]*self.observations[1][0]) + (self.observations[1][1]*self.observations[1][1])
        return rew
    
    # def joint_torques_l2_np(self):
    #     """
    #     Penalize joint torques using an L2-squared cost.

    #     Args:
    #         joint_ids (list[int] or None): Indices of joints to include in the penalty.
    #                                     If None, all joints are included.

    #     Returns:
    #         float: L2 penalty value (sum of squared torques).
    #     """
    #     # Extract joint torques
    #     torques = np.array(self.observations[10])  # shape: (num_joints,)

    #     # Select subset if joint_ids provided
    #     # if joint_ids is not None:
    #     #     torques = torques[joint_ids]

    #     # Compute L2-squared penalty
    #     penalty = np.sum(np.square(torques))

    #     return penalty
    



    # reward to lift feet at certain height

    # def reward_feet_air_height(self, 
    #                    desired_air_height=0.12,   # REDUCED from 0.2 to 8cm (more realistic)
    #                    max_air_height=0.2,       # REDUCED from 0.25
    #                    symmetric_bonus=True,
    #                    stationary_threshold=0.1):
        
    #     base_height = self.data.qpos[2]  # Base height
    #     foot_names_ = ["fl", "fr", "hl", "hr"]
    #     foot_heights = []
        
    #     for i in range(4):
    #         # Get actual foot height relative to base
    #         foot_body_id = self.model.body(foot_names_[i] + "_lleg").id
    #         foot_height = self.data.xpos[foot_body_id][2] - base_height
    #         foot_heights.append(foot_height)
        
    #     foot_heights = np.array(foot_heights)
        
    #     # Stronger reward for proper lifting
    #     height_reward = np.exp(-((foot_heights - desired_air_height) ** 2) / (2 * (0.02 ** 2)))  # Tighter
        
    #     # Penalize if too high (jumping)
    #     too_high_penalty = np.clip((foot_heights - max_air_height), 0, None)
    #     too_high_penalty = np.sum(too_high_penalty) * -0.2  # Increased penalty
        
    #     # Encourage alternating lift
    #     rhythmic_penalty = 0
    #     num_in_air = np.sum(self.feet_contacts == 0)
    #     if num_in_air == 0:  # All feet on ground - bad for walking!
    #         rhythmic_penalty = -0.2
    #     elif num_in_air >= 2:  # Good - at least 2 feet lifting
    #         rhythmic_bonus = 0.1
    #     else:
    #         rhythmic_bonus = 0.0

    #     # Base reward
    #     reward = np.mean(height_reward) + too_high_penalty + rhythmic_penalty
        
    #     # Only apply if moving
    #     if self.observations[0][0]/2.0 < stationary_threshold:
    #         reward = 0.0

    #     return float(reward)

    def reward_feet_air_height_backward(self, desired_air_height=0.08):
        """
        Optimized for backward walking - lower, more efficient foot lift
        """
        base_height = self.data.qpos[2]
        foot_names_ = ["fl", "fr", "hl", "hr"]
        foot_heights = []
        
        for i in range(4):
            foot_body_id = self.model.body(foot_names_[i] + "_lleg").id
            foot_height = self.data.xpos[foot_body_id][2] - base_height
            foot_heights.append(foot_height)
        
        foot_heights = np.array(foot_heights)
        
        # For backward walking, we want efficient, low foot lift
        total_reward = 0.0
        feet_in_swing = 0
        
        for i in range(4):
            if self.feet_contacts[i] == 0:  # Swing phase
                feet_in_swing += 1
                # Reward efficient backward stepping (lower than forward)
                height_error = abs(foot_heights[i] - desired_air_height)
                swing_reward = np.exp(-height_error / 0.02)
                total_reward += swing_reward
        
        if feet_in_swing > 0:
            total_reward = total_reward / feet_in_swing
        
        # Only reward if actually moving backward
        if self.observations[0][0] / 2.0 > -0.1:  # Not moving backward enough
            total_reward = 0.0
            
        return total_reward

    def reward_gait_quality_backward(self):
        """
        Gait pattern optimized for backward walking
        In backward walking, hind legs lead the motion
        """
        contacts = self.feet_contacts
        reward = 0.0
        
        # In backward walking, hind legs should initiate movement
        # Check if hind legs are lifting more frequently
        hind_legs_in_air = contacts[2] == 0 or contacts[3] == 0
        front_legs_in_air = contacts[0] == 0 or contacts[1] == 0
        
        if hind_legs_in_air:
            reward += 0.5  # Reward hind leg movement (important for backward)
        
        # Still want alternating diagonal pattern
        if contacts[0] != contacts[3]:  # FL vs HR
            reward += 0.25
        if contacts[1] != contacts[2]:  # FR vs HL  
            reward += 0.25
        
        return reward
    
    def reward_foot_lift_balanced(self,
                              desired_air_height=0.10,
                              stance_height_thresh=0.03):
        base_h = self.data.qpos[2]
        reward = 0.0

        # Foot body names
        foot_names = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]

        for i, name in enumerate(foot_names):
            foot_id = self.model.body(name).id

            foot_h = self.data.xpos[foot_id][2] - base_h
            in_contact = self.feet_contacts[i]

            if in_contact:  
                # stance → penalize lift above threshold
                if foot_h > stance_height_thresh:
                    reward -= (foot_h - stance_height_thresh) * 5.0
            else:
                # swing → reward lift up to target height
                diff = abs(foot_h - desired_air_height)
                reward += np.exp(-diff / 0.03)

        return reward / 4.0


    # def reward_leg_alignment(self):
    #     """
    #     Reward proper leg alignment (legs moving parallel to body)
    #     Penalize splayed/outward-angled legs
    #     """
    #     reward = 0.0
        
    #     # Get hip joint angles - these control leg splay
    #     # Assuming your joint order is: [fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, ...]
    #     hip_abduction_angles = [
    #         self.data.qpos[7],   # fl_hx - front left hip abduction
    #         self.data.qpos[10],  # fr_hx - front right hip abduction  
    #         self.data.qpos[13],  # hl_hx - hind left hip abduction
    #         self.data.qpos[16]   # hr_hx - hind right hip abduction
    #     ]
        
    #     # Ideal hip abduction is around 0 radians (legs straight down)
    #     # Penalize large outward angles (positive or negative)
    #     for angle in hip_abduction_angles:
    #         alignment_error = abs(angle)  # Distance from ideal 0 angle
    #         # Reward for small angles (good alignment)
    #         alignment_reward = 1.0 - min(alignment_error / 0.3, 1.0)  # 0.3 rad ~ 17 degrees
    #         reward += alignment_reward
        
    #     return reward / 4.0  # Average over 4 hips


    # Reward legs staying aligned under the body (not protruding out)
    def reward_leg_alignment(self):
        hip_indices = [7, 10, 13, 16]  # fl_hx, fr_hx, hl_hx, hr_hx
        
        reward_sum = 0.0
        for idx in hip_indices:
            angle = self.data.qpos[idx]
            alignment_error = abs(angle)
            alignment_reward = 1.0 - min(alignment_error / 0.25, 1.0)
            reward_sum += max(alignment_reward, 0.0)
        
        return reward_sum / 4.0

    # Feet sliding penalty
    def feet_slide_np(self):
        penalty = 0.0
        for i in range(4):
            if self.feet_contacts[i] > 0: 
                if hasattr(self.foot_vels, '__len__') and len(self.foot_vels) > 4:
                    foot_vel = self.foot_vels[i]
                    horz_vel = abs(foot_vel)
                else:
                    horz_vel = abs(self.foot_vels[i])
                
                if horz_vel > 0.1:
                    penalty += min(horz_vel * 0.5, 2.0)  
        
        return penalty
    
    def terminate(self):
        if self.observations[2] > 0.35 or self.observations[3] > 0.35 or self.observations[9] > 1.0:
            self.terminate_1 = True


        if self.observations[2] > 0.35 or self.observations[3] > 0.35  or self.current_step >= self.num_steps_per_ep or self.observations[9] > 1.0:
            if self.current_step >= self.num_steps_per_ep:
                print("Episode ended")
            if self.observations[2] > 0.35 or self.observations[3] > 0.35:
                print("roll or pitch")
            if self.observations[9] > 1.0:
                # print(self.observations[9])
                print("height")
            self.current_step = 0
            return True
        return False
    

    # penalizing bigger changes from default position
    def posture_penalty(self):
        q = self.data.qpos[7:19]
        default = np.array(list(default_joint_angles.values()))
        diff = q - default
        return np.sum(diff**2)
    
    # to prevent jump kind of walking
    def symmetry_penalty(self):
        q = self.data.qpos[7:19]

        # Map: fl_hx, fl_hy, fl_kn, fr_hx...
        fl = q[0:3]
        fr = q[3:6]
        hl = q[6:9]
        hr = q[9:12]

        penalty = 0
        penalty += np.sum((fl - fr)**2)
        penalty += np.sum((hl - hr)**2)
        return penalty
    
    def reward_gait_quality_forward(self):
        contacts = self.feet_contacts
        reward = 0.0
        
        # In trot gait: diagonal legs OUT of phase
        if contacts[0] != contacts[3]:  
            reward += 0.25
        if contacts[1] != contacts[2]:  
            reward += 0.25

        # Left-right legs IN phase opposition
        if contacts[0] == contacts[2]:  # Both left legs move together
            reward += 0.25
        if contacts[1] == contacts[3]:  # Both right legs move together
            reward += 0.25

        return reward


    def rewards(self, current_action):
        # === DIRECTIONAL REWARDS & PENALTIES ===
        current_lin_x = self.observations[0][0] / 2.0
        current_lin_y = self.observations[0][1] / 2.0
        current_ang_vel = self.observations[1][2]
        desired_lin_x = self.current_cmd[0]
        
        # 1. MAIN BACKWARD REWARD (STRONG)
        if current_lin_x < -0.1:  # Moving backward
            backward_reward = 4.0 * min(abs(current_lin_x) / 1.0, 1.0)  # Increased from 3.0
        else:
            backward_reward = 0.0
        
        # 2. BALANCED PENALTIES (REDUCED)
        wrong_direction_penalty = -1.0 if current_lin_x > 0.05 else 0.0  # Reduced from -3.0
        lateral_penalty = -0.5 * min(abs(current_lin_y) / 0.2, 1.0)      # Reduced from -2.0
        angular_penalty = -0.5 * min(abs(current_ang_vel) / 0.3, 1.0)    # Reduced from -2.0
        
        # 3. SPEED MATCHING
        if desired_lin_x < 0 and current_lin_x < 0:
            speed_match = 1.0 - min(abs(desired_lin_x - current_lin_x) / 0.5, 1.0)
            speed_reward = 2.0 * speed_match  # Increased from 1.0
        else:
            speed_reward = 0.0
        
        # === GAIT QUALITY REWARDS ===
        foot_lift_reward = 3.0 * self.reward_feet_air_height_backward()
        gait_quality_reward = 2.0 * self.reward_gait_quality_backward()  # Increased from 1.5
        leg_alignment_reward = 1.0 * self.reward_leg_alignment()
        
        # === PENALTIES ===
        sliding_penalty = -0.1 * min(self.feet_slide_np() / 5.0, 1.0)
        action_penalty = -0.001 * min(self._reward_action_rate(current_action) / 10.0, 1.0)
        
        total_reward = (
            backward_reward +           # [0, 4.0] - INCREASED
            speed_reward +              # [0, 2.0] - INCREASED
            foot_lift_reward +          # [0, 3.0]
            gait_quality_reward +       # [0, 2.0] - INCREASED
            leg_alignment_reward +      # [0, 1.0]
            wrong_direction_penalty +   # [-1.0, 0] - REDUCED
            lateral_penalty +           # [-0.5, 0] - REDUCED
            angular_penalty +           # [-0.5, 0] - REDUCED
            sliding_penalty +           # [-0.1, 0]
            action_penalty              # [-0.1, 0]
        )
        
        return total_reward