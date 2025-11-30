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
        # data.xpos[body_id] gives [x, y, z] of body in world frame
        foot_positions.append(data.xpos[body_id][2] - 0.2)

    return np.array(foot_positions)



class SpotEnv:
    def __init__(self, num_obs = 52, num_actions = 12, num_commands = 4, show_viewer=True, device="cuda", num_steps_per_ep = 2000, mode = 'sample'):
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
        self.mode  = mode
        # self.hl_commands = ["forward", "left", "right", "stop"]
        # self.start_pos = self.data.qpos[0:3].copy()  # x, y, z
        # self.target_distance = 2

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
        self.data.qpos[19:24] = arm_folded.copy()
        joint_angles = list(default_joint_angles.values())
        # print(joint_angles)
        if self.mode == 'sample':
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
        if self.mode == 'sample':
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
            g_body
        ]

        obs_flatten = np.concatenate([
            np.ravel(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
            for x in self.observations
        ])

        start_time = self.data.time
        timeout_ = 0 

        while True:
            self.feet_contacts = get_feet_contact(self.model, self.data, self.foot_names)
            check = np.array([1,1,1,1])
            if np.array_equal(self.feet_contacts,check) :
                break
            mu.mj_step(self.model, self.data)

            if self.data.time - start_time > 2.0:
                print("Timeout: robot did not settle within 2 seconds")
                timeout_ = 1
                break
    

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
        # vx = np.random.uniform(0, max_forward_speed)  # forward speed
        vx = 0.0
        vy = 0.0                                        # no lateral movement
        wz = np.random.uniform(-max_yaw_rate, max_yaw_rate)                                        # no rotation

        self.target_distance = np.random.uniform(0.0, 3.0)
        # print(vx)

        self.current_cmd = np.array([vx, vy, wz])
        return self.current_cmd
    

    def give_vel_command(self,ang):
        self.current_cmd = [0,0,ang]

    def action_dims(self):
        return self.num_actions
    

    def obs_dims(self):
        return self.num_obs
    





    def step(self,actions_motor_pos,view=False):        # step the environment by providing the control torques
        # self.show_viewer = view
        # motor_torques = self.position_to_torquePD(actions_motor_pos)
        # self.data.ctrl[:12] = motor_torques
        def_pos = list(default_joint_angles.values())
        new_pos = []
        for i in range(len(def_pos)):
            if actions_motor_pos[i] > 100:
                actions_motor_pos[i] = 100
            elif actions_motor_pos[i] < -100:
                actions_motor_pos[i] = -100
            new_pos.append(0.25*actions_motor_pos[i] + self.default_pos[i])

        # self.data.ctrl[:12] = 
        if self.current_cmd[2] == 0.0:
            self.data.ctrl[:12] = def_pos
        else:
            self.data.ctrl[:12] = new_pos
        # self.data.ctrl[:12] = new_pos
        self.data.qpos[19:24] = arm_folded.copy()
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

        # ---- Previous action ----
        prev_action = np.asarray(actions_motor_pos)  # shape (12,)
        torques_applied = self.data.actuator_force

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
    
    def _reward_tracking_lin_vel(self):
        current_lin_x = self.observations[0][0]
        current_lin_y = self.observations[0][1]
        lin_rew = ((self.observations[4][0] - current_lin_x)*(self.observations[4][0] - current_lin_x)) + ((self.observations[4][1] - current_lin_y)*(self.observations[4][1] - current_lin_y))
        tracking_sigma = 0.25
        lin_rew = np.exp(-lin_rew/tracking_sigma)
        return lin_rew
    
    def _reward_tracking_ang_vel(self):
        current_ang_z = self.observations[1][2]
        lin_rew = ((self.observations[4][2] - current_ang_z)*(self.observations[4][2] - current_ang_z))
        tracking_sigma = 0.25
        lin_rew = np.exp(-lin_rew/tracking_sigma)
        return lin_rew
    
    def _reward_roll_pitch_(self):
        roll, pitch = self.observations[2]*180/3.14, self.observations[3]*180/3.14
        reward = (roll*roll) + (pitch*pitch)
        return reward
    
    def _reward_ang_vel_xy(self):
        rew = (self.observations[1][0]*self.observations[1][0]) + (self.observations[1][1]*self.observations[1][1])
        return rew
    
    def reward_feet_air_height(self, 
                        desired_air_height=0.15,   # More realistic height
                        max_air_height=0.25,
                        stationary_threshold=0.1):
    
        base_height = self.data.qpos[2]  # Base height from ground
        foot_names_ = ["fl", "fr", "hl", "hr"]
        foot_heights = []
        
        for i in range(4):
            # Get actual foot height relative to BASE, not ground
            foot_body_id = self.model.body(foot_names_[i] + "_lleg").id
            foot_height = self.data.xpos[foot_body_id][2] - base_height
            foot_heights.append(foot_height)
        
        foot_heights = np.array(foot_heights)
        
        # Strong reward for proper lifting during swing phase
        height_rewards = []
        for i in range(4):
            if self.feet_contacts[i] == 0:  # Only reward lifting during swing
                height_error = abs(foot_heights[i] - desired_air_height)
                height_reward = np.exp(-height_error / 0.05)  # Tight reward around target
                height_rewards.append(height_reward)
            else:  # Stance phase - penalize lifting
                if foot_heights[i] > 0.05:
                    height_rewards.append(-0.5)
                else:
                    height_rewards.append(0.0)
        
        reward = np.mean(height_rewards) if height_rewards else 0.0
        
        # Only apply if actually trying to turn
        if abs(self.current_cmd[2]/0.25) < stationary_threshold:
            reward = 0.0
            
        return reward


    def feet_slide_np(self):
        penalty = 0.0
        for i in range(4):
            if self.feet_contacts[i] > 0:  # Only feet in contact
                if hasattr(self.foot_vels[i], '__len__') and len(self.foot_vels[i]) > 1:
                    # Use full velocity vector
                    horz_vel = np.linalg.norm(self.foot_vels[i][:2])
                else:
                    # Single value
                    horz_vel = abs(self.foot_vels[i])
                
                # Progressive penalty - more sliding = higher penalty
                if horz_vel > 0.05:
                    penalty += horz_vel * 2.0  # Strong penalty for sliding
        
        return penalty
    
    def reward_gait_quality(self):
        reward = 0.0
        
        
        contacts = self.feet_contacts
        
        # Check diagonal synchronization
        if contacts[0] == contacts[3]:  # FL and HR sync
            reward += 0.25
        if contacts[1] == contacts[2]:  # FR and HL sync
            reward += 0.25
            
        # Check opposite legs are out of phase
        if contacts[0] != contacts[1]:  # Front left vs right
            reward += 0.25
        if contacts[2] != contacts[3]:  # Hind left vs right
            reward += 0.25
        
        return reward
    
    def posture_penalty(self):
        q = self.data.qpos[7:19]
        default = np.array(list(default_joint_angles.values()))
        diff = q - default
        return np.sum(diff**2)
    
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

    def reward_leg_alignment(self):
        reward = 0.0
        
        hip_abduction_angles = [
            self.data.qpos[7],   
            self.data.qpos[10],  
            self.data.qpos[13],  
            self.data.qpos[16]   
        ]
        
        # Penalize large outward angles (positive or negative)
        for angle in hip_abduction_angles:
            alignment_error = abs(angle) 
            alignment_reward = 1.0 - min(alignment_error / 0.3, 1.0) 
            reward += alignment_reward
        
        return reward / 4.0  
    
    def reward_leg_posture(self):
        
        reward = 0.0
        
        # Hip abduction angles (these control leg splay)
        hip_abduction_angles = [
            abs(self.data.qpos[7]),   # fl_hx
            abs(self.data.qpos[10]),  # fr_hx  
            abs(self.data.qpos[13]),  # hl_hx
            abs(self.data.qpos[16])   # hr_hx
        ]
        
        # Knee angles (prevent over-bent knees)
        knee_angles = [
            abs(self.data.qpos[9]),   # fl_kn  
            abs(self.data.qpos[12]),  # fr_kn
            abs(self.data.qpos[15]),  # hl_kn
            abs(self.data.qpos[18])   # hr_kn
        ]
        
        # Reward natural hip angles (legs under body, not splayed)
        for angle in hip_abduction_angles:
            # Ideal is near 0 radians, penalize angles > 0.1 rad (~6 degrees)
            if angle < 0.1:
                reward += 0.25  # Good posture
            elif angle < 0.2:
                reward += 0.1   # Acceptable
            else:
                reward -= 0.1   # Bad - splayed
        
        # Reward proper knee bend (not too straight, not too bent)
        for angle in knee_angles:
            # Ideal knee angle around 1.0-1.5 radians for walking
            if 0.8 < angle < 1.7:
                reward += 0.1   # Good knee bend
            elif angle > 2.0:
                reward -= 0.1   # Over-bent knees
        
        return reward


    
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
    
    
    def rewards(self, current_action):
        # 1) Yaw-rate tracking (main term)
        current_ang_vel = self.observations[1][2]  # undo *0.25
        desired_ang_vel = self.current_cmd[2]             # already in rad/s
        ang_vel_error = abs(desired_ang_vel - current_ang_vel)
        yaw_track = np.exp(- (ang_vel_error ** 2) / (2 * 0.3 ** 2))  # in [0,1]

        # 2) Penalize forward motion but with deadzone
        v_xy = np.linalg.norm(self.observations[0][:2] / 2.0)  # undo *2
        if v_xy < 0.05:
            forward_penalty = 0.0
        else:
            forward_penalty = min((v_xy - 0.05) / 0.25, 1.0)   # in [0,1]

        # 3) Sliding penalty (softened)
        slide_raw = self.feet_slide_np()   # assume updated version with 0.1 threshold
        slide_penalty = min(slide_raw / 1.0, 1.0)             # normalize to [0,1]

        # 4) Orientation penalty
        tilt = self._reward_roll_pitch_()   # deg^2
        tilt_penalty = min(tilt / 200.0, 1.0)  #  ~≤15° => below saturation

        # 5) Helper terms (clamped non-negative)
        foot_lift = max(self.reward_feet_air_height(), 0.0)   # clamp to ≥ 0
        foot_lift = min(foot_lift, 1.0)

        gait_quality = self.reward_gait_quality()             # in [0,1] roughly
        gait_quality = np.clip(gait_quality, 0.0, 1.0)
        p_pen = min(self.posture_penalty() / 10.0, 1.0)
        sym_pen = min(self.symmetry_penalty() / 5.0, 1.0)

        leg_alignment_reward = 2.0 * self.reward_leg_alignment()

        # posture_reward = 1.5 * self.reward_leg_posture()      # Strong posture reward
    # height_reward = 0.8 * self.reward_body_height()       # B   

        # Combine — all weights small / balanced
        reward = (
            # posture_reward +
            leg_alignment_reward +
            2.0 * yaw_track +         # [0, 2]
            0.5 * foot_lift +         # [0, 0.5]
            0.5 * gait_quality -      # [-0.5, 0]
            0.8 * forward_penalty -   # [-0.8, 0]
            0.8 * slide_penalty -     # [-0.8, 0]
            0.5 * tilt_penalty -       # [-0.5, 0]
            0.5 * p_pen -       # posture deviation penalty
            0.3 * sym_pen
        )

        return reward
