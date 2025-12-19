from spot_env_turn import SpotEnv_Turn
from spot_env import SpotEnv_straight
import numpy as np
from ppo import PPO
from spot_share_env import SpotEnv_Common
from gymnasium import spaces
import gymnasium as gym
import time

class HierarchicalSpotEnv(gym.Env):
    def __init__(self, straight_policy_path, turn_policy_path):

        super().__init__()
        # Load low-level policies
        # self.straight_policy = self.load_policy(straight_policy_path, is_turning=False)
        # self.turn_policy = self.load_policy(turn_policy_path, is_turning=True)
        
        # Door detection (your existing module)
        # self.door_detector = YourDoorDetectionModule()
        
        # Environment state
        self.total_steps = 0
        self.max_episode_steps = 10000
        # self.spot_env_yaw = SpotEnv_Turn(mode='command')
        # self.spot_env_lin  = SpotEnv_straight(mode='command')
        self.pos_or_command = [0.0,0.0,0.0]
        self.observations = []

        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_shape,), 
            dtype=np.float32
        )

        # self.action_space = spaces.Box(
        #     low=np.array([0]),     # policy, straight_vel, turn_vel
        #     high=np.array([1]), 
        #     dtype=np.float32
        # )

        self.action_space = spaces.Discrete(3)


        self.current_action = 0
        self.current_policy = SpotEnv_Common(mode='command')
    
        action_std = 0.5            # set same std for action distribution which was used while saving
        K_epochs = 20               # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.99                # discount factor

        lr_actor = 0.001           # learning rate for actor
        lr_critic = 0.003           # learning rate for critic

        #####################################################

        # state space dimension
        state_dim = self.current_policy.obs_dims()

        # action space dimension
        action_dim = self.current_policy.action_dims()
        # initialize a PPO agent
        self.ppo_agent_straight = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std)
        
        self.ppo_agent_turn = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std)

        # checkpoint_path = "/home/prashanth/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_straight_2.pth"
        # checkpoint_path_t = "/home/prashanth/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_turn_4_final.pth"
        self.ppo_agent_straight.load(straight_policy_path)
        self.ppo_agent_turn.load(turn_policy_path)
        self.ppo_agent = self.ppo_agent_straight
        self.terminate = False
        self.observation_len = 13

        self.aligned_to_goal = False
        self.reached_position = False 
        self.mission_complete = False
        
        # Previous values for improvement tracking
        self.prev_position = None
        self.prev_dir_err = None
        self.prev_yaw_err = None
        self.prev_pos_err = None
        self.steps_since_switch = 0
        self.last_action = None
        self.switch_cooldown = 5   # number of steps to enforce STOP
        self.goal_reached = False
        self.best_pos_err = None
        self.best_yaw_err = None
        self.prev_action = None
        self.phase = 0
        self.turn_streak = 0
        self.door_pos = None
        self.first_att = 0


    def _get_obs_shape(self):
        # Get observation shape by sampling reset
        return 4
    
    def apply_high_level_action(self, action):
        """
        action: integer ∈ {0,1,2,3}
        returns: (forward_vel, turn_vel)
        """

        FORWARD_SPEED = 0.7
        TURN_SPEED = 0.4

        if action == 0:      # TURN LEFT
            return 0, +TURN_SPEED
        
        elif action == 1:    # TURN RIGHT
            return 0.0, -TURN_SPEED
        
        elif action == 2:    # STRAIGHT
            return 1.0, FORWARD_SPEED
        
        else:                # STOP
            return 0.0, 0.0

        
    def reset(self,seed=None, options=None):

        # self.pos_or_command[0:2] = np.array([
        #         np.random.uniform(7.0, 13.0),    # x: 7-13m (6m range)
        #         np.random.uniform(-7.5, -3.5),   # y: -5.5 to -3.5m (2m range)
        # ])

        self.prev_action = None


        self.aligned_to_goal = False
        self.reached_position = False
        self.mission_complete = False
        self.goal_reached = False
        self.prev_position = None
        self.prev_dir_err = None
        self.prev_yaw_err = None
        self.prev_pos_err = None
        self.best_pos_err = None
        self.best_yaw_err = None
        self.last_action = None
        self.turn_streak = 0
        #self.state, _ = self.current_policy.reset(self.pos_or_command[0:2])
        self.door_pos = self.current_policy.door_detect_env()
        
        if self.first_att == 0:
           self.state, _ = self.current_policy.reset(self.pos_or_command[0:2])
           self.first_att+=1
        
        if self.door_pos is None:
            self.pos_or_command[0:2] = np.array([
                    np.random.uniform(7.0, 13.0),    # x: 7-13m (6m range)
                    np.random.uniform(-7.5, -3.5),   # y: -5.5 to -3.5m (2m range)
            ])

            #self.pos_or_command[0:2] = [12.426,-1.01]
            #yaw = np.random.uniform(-1.57, 1.57)
            yaw = 1.57
            self.pos_or_command[2] = yaw
        else:
            #self.door_pos = self.current_policy.door_detect_env()
            if self.door_pos is not None:
                self.pos_or_command[0:2] = self.door_pos[0:2]
                yaw = 1.57
                self.pos_or_command[2] = 1.57
                #self.pos_or_command[2] = self.door_pos[2]
                #self.pos_or_command[1] = self.pos_or_command[1]
                #self.pos_or_command[0] = self.pos_or_command[0] + 1
        
        #self.state, _ = self.current_policy.reset(self.pos_or_command[0:2])
                

        
        
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        # self.pos_or_command[0:2] = np.array([
        #     np.random.uniform(5.0, 20.0),   # x offset
        #     np.random.uniform(-7.5, -3.5),   # y offset
        #     # 0.35                            # z height above ground
        # ])
        
        # self.pos_or_command[0:2] = np.array([
        #         np.random.uniform(7.0, 13.0),    # x: 7-13m (6m range)
        #         np.random.uniform(-5.5, -3.5),   # y: -5.5 to -3.5m (2m range)
        # ])

        #yaw = np.random.uniform(-1.57, 1.57)
        #self.pos_or_command[2] = yaw

        current_pos = self.current_policy.get_current_pose()

        current_x, current_y = current_pos[0], current_pos[1]
        euclidean_distance = np.sqrt((self.pos_or_command[0] - current_x)**2 + (self.pos_or_command[1] - current_y)**2)
        
        # Calculate yaw error
        current_yaw = current_pos[2]
        yaw_error = abs(self.pos_or_command[2] - current_yaw)


        cx, cy, cyaw = self.current_policy.get_current_pose()
        tx, ty, tyaw = self.pos_or_command

        dx = tx - cx
        dy = ty - cy

        distance_error = np.sqrt(dx*dx + dy*dy)

        # Compute angle error
        goal_direction = np.arctan2(dy, dx)
        direction_error = self.wrap_angle(goal_direction - cyaw)

        final_yaw_error = self.wrap_angle(tyaw - cyaw)

        self.observations = np.array([
            dx,
            dy,
            distance_error,
            direction_error
        ], dtype=np.float32)

        self.terminate = False

        obs_flatten = np.concatenate([
            np.ravel(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
            for x in self.observations
        ])

        return obs_flatten.astype(np.float32), {}
    
    def wrap_angle(self,angle):
        """Wrap angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_direction_error(self,current_x, current_y, current_yaw,
                                target_x, target_y):
        """
        Returns:
            direction_error (radians, absolute, between 0 and pi)
            goal_direction (the desired heading)
        """
        # Vector from robot → goal
        dx = target_x - current_x
        dy = target_y - current_y

        # Heading angle toward goal
        goal_direction = np.arctan2(dy, dx)

        # Error between robot yaw and goal direction
        error = self.wrap_angle(goal_direction - current_yaw)

        return abs(error), goal_direction


    
    def step(self, high_level_action):
        """
        high_level_action format: {
            'policy_type': 0 or 1,  # 0=straight, 1=turn
            'velocity': float,      # -1.0 to 1.0
            'duration': int         # 1 to 10 steps
        }
        """

        # print(f"High-level action: {high_level_action}")
        # print(f"Current pose: {self.current_policy.get_current_pose()}")
        # print(f"Target: {self.pos_or_command}")

                # print(f"High-level action: {high_level_action}")
        # print(f"Current pose: {self.current_policy.get_current_pose()}")
        # print(f"Target: {self.pos_or_command}")
        #if self.door_pos is None:
        #    self.door_pos = self.current_policy.door_detect_env()
        #    if self.door_pos is not None:
        #        self.pos_or_command[0:2] = self.door_pos[0:2]
        ##        self.pos_or_command[2] = 1.57
         #       #self.pos_or_command[2] = self.pos_or_command[2]
         #       self.pos_or_command[1] = self.pos_or_command[1]
         #       #self.pos_or_command[0] = self.pos_or_command[0] - 2

        if self.last_action is not None:
            switching = (high_level_action != self.last_action)

            # Only activate cooldown if switching TO a motion skill (0,1,2)
            if switching and high_level_action in [0, 1, 2]:
                self.steps_since_switch = self.switch_cooldown

        self.last_action = high_level_action

        # Enforce STOP during cooldown
        if self.steps_since_switch > 0:
            safe_action = 3      # STOP
            self.steps_since_switch -= 1
        else:
            safe_action = high_level_action

        if self.goal_reached == True:
            safe_action = 3

        action_ , vel = self.apply_high_level_action(safe_action)

        if action_  == 1.0:
            pol = True
        else:
            pol = False
        # Parse high-level action
        if pol:
            self.ppo_agent = self.ppo_agent_straight
            self.current_policy.give_vel_command(vel,pol)
            # self.state = self.st_obs
        else:
            self.ppo_agent = self.ppo_agent_turn
            self.current_policy.give_vel_command(vel,pol)
            # self.state = self.tr_obs
        
        # self.current_policy.give_vel_command(high_level_action[1],pol)
        action = self.ppo_agent.select_action(self.state)

        if pol:
            self.state, reward, done, _ =self.current_policy.step_straight(action)
        else:
            self.state, reward, done, _ =self.current_policy.step_turn(action)


        current_pos = self.current_policy.get_current_pose()

        current_x, current_y = current_pos[0], current_pos[1]
        euclidean_distance = np.sqrt((self.pos_or_command[0] - current_x)**2 + (self.pos_or_command[1] - current_y)**2)
        
        # Calculate yaw error
        current_yaw = current_pos[2]
        yaw_error = abs(self.pos_or_command[2] - current_yaw)


        cx, cy, cyaw = self.current_policy.get_current_pose()
        tx, ty, tyaw = self.pos_or_command

        dx = tx - cx
        dy = ty - cy

        distance_error = np.sqrt(dx*dx + dy*dy)

        # Compute angle error
        goal_direction = np.arctan2(dy, dx)
        direction_error = self.wrap_angle(goal_direction - cyaw)

        final_yaw_error = self.wrap_angle(tyaw - cyaw)

        self.observations = np.array([
            dx,
            dy,
            distance_error,
            direction_error,
        ], dtype=np.float32)


        # self.observations = [
        #     self.pos_or_command[0:2],
        #     self.pos_or_command[2],
        #     euclidean_distance,
        #     yaw_error,
        #     current_pos[0:2],
        #     current_yaw,
        #     current_pos[0] - self.pos_or_command[0],
        #     current_pos[1] - self.pos_or_command[1],
        #     high_level_action
        # ]

        obs_flatten = np.concatenate([
            np.ravel(x) if isinstance(x, (list, np.ndarray)) else np.array([x])
            for x in self.observations
        ])


        curr_rewards, pos_err, yaw_err = self._compute_high_level_reward(high_level_action)
        # print(f"Reward: {reward}, Pos error: {pos_err}, Yaw error: {yaw_err}")
        complete, self.goal_reached = self._check_termination(pos_err,yaw_err)

        if self.goal_reached:
            self.current_policy.give_vel_command(0,True)
            action = self.ppo_agent.select_action(self.state)
            self.state, reward, done, _ =self.current_policy.step_straight(action)

        if self.terminate == True:
            curr_rewards-=100


        self.total_steps+=1
        
        #print(final_yaw_error)
        if pos_err < 1.0 and abs(yaw_error) > 0.4:
        #if False:
            while True:
                print("in check")
                cx, cy, cyaw = self.current_policy.get_current_pose()
                tyaw = self.pos_or_command[2]
            
                #signed_yaw_err = np.arctan2(np.sin(tyaw - cyaw), np.cos(tyaw - cyaw))
                signed_yaw_err = (tyaw - cyaw)
                print(cyaw,tyaw,signed_yaw_err)
                if signed_yaw_err > 0 :
                    self.ppo_agent = self.ppo_agent_turn
                    self.current_policy.give_vel_command(0.4,False)
                    action = self.ppo_agent.select_action(self.state)
                    self.state, reward, done, _ =self.current_policy.step_turn(action)
                elif signed_yaw_err < 0:
                    self.current_policy.give_vel_command(-0.4,False)
                    self.ppo_agent = self.ppo_agent_turn
                    action = self.ppo_agent.select_action(self.state)
                    self.state, reward, done, _ =self.current_policy.step_turn(action)
				
                if abs(signed_yaw_err) < 0.2:
                    self.current_policy.give_vel_command(0.0,True)
                    self.ppo_agent = self.ppo_agent_straight
                    action = self.ppo_agent.select_action(self.state)
                    self.state, reward, done, _ =self.current_policy.step_straight(action)
                    complete = True
                    self.total_steps = 0
                    break					
					

        return (
                    obs_flatten.astype(np.float32), 
                    float(curr_rewards), 
                    complete,    # Your goal_reached or terminate condition
                    False,         # truncated (time limit) - set to False or your condition
                    {}             # info dict
                )

    def to_continuous_angle(self,current_angle, reference_angle):
            """Convert angle to continuous form (can be >2π or < -2π)"""
            # Find difference and adjust to make continuous
            diff = current_angle - reference_angle
            # Add/subtract 2π to make diff in reasonable range
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            # Return continuous angle
            return reference_angle + diff

    def _compute_high_level_reward(self, action):
        TURN_RIGHT, TURN_LEFT, STRAIGHT = 0, 1, 2

        # ----------------------------
        # 1. Robot pose + goal
        # ----------------------------
        x, y, yaw = self.current_policy.get_current_pose()
        tx, ty, tyaw = self.pos_or_command

        pos_err = np.hypot(tx - x, ty - y)

        # ----------------------------
        # 2. Direction error
        # ----------------------------
        goal_dir = np.arctan2(ty - y, tx - x)

        signed_dir_err = np.arctan2(
            np.sin(goal_dir - yaw),
            np.cos(goal_dir - yaw)
        )
        dir_err = abs(signed_dir_err)

        # ----------------------------
        # 3. Improvement tracking
        # ----------------------------
        if not hasattr(self, "prev_pos_err") or self.prev_pos_err is None:
            self.prev_pos_err = pos_err
            self.prev_dir_err = dir_err
            self.prev_action = action

        dist_improve = self.prev_pos_err - pos_err
        dir_improve  = self.prev_dir_err - dir_err

        reward = 0.0

        # =======================================================
        # 4. FIXED ACTION LOGIC (turn only when needed)
        # =======================================================
        ALIGN_THRESHOLD = 0.2   # rad (~35°)

        correct_turn = TURN_RIGHT if signed_dir_err > 0 else TURN_LEFT

        # -------------------------------------------------------
        # TURNING BEHAVIOR
        # -------------------------------------------------------
        if action in (TURN_LEFT, TURN_RIGHT):

            # ❌ turning while aligned → bad
            if dir_err < ALIGN_THRESHOLD:
                reward -= 3.0

            # ✔ turning when misaligned
            else:
                reward += 1.0
                if action == correct_turn and dir_improve > 0:
                    reward += 1.0

            # ❌ oscillation between left ↔ right
            if self.prev_action in (TURN_LEFT, TURN_RIGHT) and action != self.prev_action:
                reward -= 1.0

        # -------------------------------------------------------
        # STRAIGHT BEHAVIOR
        # -------------------------------------------------------
        if action == STRAIGHT:

            # ❌ too misaligned → do NOT go straight
            if dir_err > ALIGN_THRESHOLD:
                reward -= 2.0

            # ✔ aligned → go straight
            else:
                reward += 2.0

            # ✔ reward forward progress
            if dist_improve > 0:
                reward += 40.0 * dist_improve
            else:
                reward -= 15.0 * (-dist_improve)

        # -------------------------------------------------------
        # Anti-stall penalty
        # -------------------------------------------------------
        if dist_improve == 0 and dir_improve == 0:
            reward -= 0.05

        # -------------------------------------------------------
        # Completion condition
        # -------------------------------------------------------
        if pos_err < 1.0:
            reward += 300.0
            self.mission_complete = True

        # Small time decay
        reward -= 0.01

        # Update history
        self.prev_pos_err = pos_err
        self.prev_dir_err = dir_err
        self.prev_action = action

        return reward, pos_err, dir_err

    
    def _check_termination(self, position_error, yaw_error):

        goal_reached = False
        #if yaw_error != None:
        if position_error < 1.0: 
            #and yaw_error < 0.4:
           goal_reached = True
        
        # print("0",self.current_policy.current_cmd[0])
        # print("2",self.current_policy.current_cmd[2])

        if self.current_policy.terminate_1 == True or self.current_policy.current_cmd[0] < 0.0 or self.current_policy.current_cmd[0] > 1.3 or self.current_policy.current_cmd[2] < -0.8 or self.current_policy.current_cmd[2] > 0.8:
            self.terminate = True

        # print(self.total_steps)

        if self.total_steps >= self.max_episode_steps or self.terminate or goal_reached:
            self.total_steps = 0
            # print("episode end hrl")
            return True, goal_reached
        # print("no")
        return False, goal_reached
    
