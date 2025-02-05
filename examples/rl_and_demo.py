"""
To run an rlenv using endeffector pose and collecting demonstrations.
"""
import numpy as np
from scipy.spatial.transform import Rotation   
from pyquaternion import Quaternion
import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
from rlbench.action_modes.action_mode import EEPlannerActionMode
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete, GripperJointPosition

import rlbench

action_mode = EEPlannerActionMode()
env = gym.make('rlbench/reach_target-state-v0', render_mode="rgb_array", action_mode=action_mode, max_episode_steps=200)

# create a demo
demos = env.unwrapped.get_demos(1, live_demos=True)
DEMO_NUM = 0

# reset to the begining of the demo
env.reset(options={"demo": demos[DEMO_NUM]}) 

# run the environment witht the same action as the demo
terminated = False
counter = 0
while not terminated:
    gripper_pose = demos[DEMO_NUM]._observations[counter].gripper_pose
    gripper_open = demos[DEMO_NUM]._observations[counter].gripper_open
    action = np.concatenate([gripper_pose, np.array([gripper_open])])
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step:{counter} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
    env.render()  # Note: rendering increases step time.
    counter += 1

# now just run normal RL loop
env.reset()
episodes = 1
counter = 0
terminated = False
truncated = False
for i in range(episodes):
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, _ = env.step(action)
        except Exception as e:
            continue
        print(f"Step:{counter} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
        env.render()  # Note: rendering increases step time.
        counter += 1
        if terminated or truncated:
            obs = env.reset()
            counter = 0

print('Done')

fps = benchmark_step(env, target_duration=10)
print(f"FPS: {fps:.2f}")


env.close()
