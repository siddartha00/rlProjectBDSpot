# Spot Quadruped RL — Policy Testing Guide

This repository contains reinforcement-learning policies for a Spot-like quadruped simulated in MuJoCo.
You can test **straight walking** or **turning-in-place** behaviors using the final trained models provided in:

```
updated_final_policies/
```

---

## Project Structure

```
.
├── spot_env.py               # Environment for straight walking
├── spot_env_turn.py          # Environment for turning
├── test.py                   # Script for running policies
├── test_command.py                   # Script for running policies
├── updated_final_policies/   # Folder containing trained models
└── README.md
```

---

# How to Use This Code

## 1. Select which behavior you want to test

Open `test_command.py` and toggle these import lines:

### Straight Walking:
```python
from spot_env import SpotEnv
# from spot_env_turn import SpotEnv
```

### Turning in Place:
```python
# from spot_env import SpotEnv
from spot_env_turn import SpotEnv
```

Just comment/uncomment based on what you want to test.

---

## 2. Set the model path

Inside `test_command.py`, locate:

```python
checkpoint_path = "PATH_TO_MODEL"
```

Replace it with the desired model from `updated_final_policies/`.

Example for straight walking:

```python
checkpoint_path = "updated_final_policies/PPO_spot_env_walking_0_0_straight_2.pth"
```

Make sure the file name matches the model you want to use.

---

## Important Note for TURNING Policy

If you are testing the **turning policy** or **straight policy**`,
you must ensure that the MuJoCo viewer is enabled.

Open **`spot_env_turn.py`** or **`spot_env.py`** and set the default value of `show_viewer`  
inside the `SpotEnv` class constructor to **True**:

```python
class SpotEnv:
    def __init__(self, num_obs=52, num_actions=12, num_commands=4, show_viewer=True, device="cuda", num_steps_per_ep=2000):
        ...
```

This ensures that the MuJoCo visualization opens automatically when testing the policy.

---

## 3. Set Velocity Command and Run

Once imports and model path are set:

Set vel variable inside the while loop of test_command.py

```bash
while True:
    vel = 0.5       # change this based on requirement (this will be angular velocity for turn and linear velocity for straight)
    env.give_vel_command(vel)
```

```bash
python test_command.py
```

This will:

- Load the appropriate Spot environment
- Load the selected trained model
- Run and visualize the robot’s motion in MuJoCo
