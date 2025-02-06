"""
To run an rlenv using endeffector pose and collecting demonstrations.
"""
import numpy as np
from pyquaternion import Quaternion
import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
from rlbench.action_modes.action_mode import EEPlannerAbsoluteActionMode, EEPlannerRelativeActionMode


import rlbench

def get_relative_pose(pose1, pose2):
    """
    Returns the relative pose from pose1 (current) to pose2 (next)
    Each pose is a 7D vector [x, y, z, qx, qy, qz, qw] with quaternion in the order of [x, y, z, w]
    """
    position1 = np.array(pose1[:3])
    position2 = np.array(pose2[:3])
    quat1 = Quaternion(pose1[-1],*pose1[3:-1])
    quat2 = Quaternion(pose2[-1], *pose2[3:-1])
    relative_position = position2 - position1
    relative_quat = quat2 * quat1.inverse
    relative_quat = np.array([relative_quat.x, relative_quat.y, relative_quat.z, relative_quat.w])
    return np.concatenate([relative_position, relative_quat])



action_mode = EEPlannerRelativeActionMode()
env = gym.make('rlbench/close_drawer-state-v0', render_mode="human", action_mode=action_mode, max_episode_steps=200)

# create a demo
print("Creating a demo")
demos = env.unwrapped.get_demos(3, live_demos=True)

print("Following the actions of the demo")
# run the environment witht the same action as the demo
for i in range(len(demos)):
    # reset to the begining of the demo
    env.reset(options={"demo": demos[i]}) 
    steps = len(demos[i]._observations)
    for step in range(steps-1):
        gripper_pose_current = demos[i]._observations[step].gripper_pose
        gripper_pose_next = demos[i]._observations[step+1].gripper_pose
        gripper_pose_relative = get_relative_pose(gripper_pose_current, gripper_pose_next)
        gripper_open = demos[i]._observations[step].gripper_open
        action = np.concatenate([gripper_pose_relative, np.array([gripper_open])])
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step:{step} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
        env.render()

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
