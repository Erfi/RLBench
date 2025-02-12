"""
To run an rlenv using endeffector pose and collecting demonstrations.
"""

import numpy as np
from pyquaternion import Quaternion
import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import rlbench


def get_relative_pose(pose1, pose2):
    """
    Returns the relative pose from pose1 (current) to pose2 (next)
    Each pose is a 7D vector [x, y, z, qx, qy, qz, qw] with quaternion in the order of [x, y, z, w]
    """
    position1 = np.array(pose1[:3])
    position2 = np.array(pose2[:3])
    quat1 = Quaternion(pose1[-1], *pose1[3:-1])
    quat2 = Quaternion(pose2[-1], *pose2[3:-1])
    relative_position = position2 - position1
    relative_quat = quat2 * quat1.inverse
    relative_quat = np.array(
        [relative_quat.x, relative_quat.y, relative_quat.z, relative_quat.w]
    )
    return np.concatenate([relative_position, relative_quat])


if False:  # USING JOINT_POSITION
    env = gym.make(
        "rlbench/close_drawer-state-v0",
        render_mode="human",
        action_type="joint_position_absolute",
        observation_type="state",
        max_episode_steps=200,
    )

    # create a demo
    print("Creating a demo")
    demos = env.unwrapped.get_demos(5, live_demos=True)

    print("Following the actions of the demo")
    # run the environment witht the same action as the demo
    for i in range(len(demos)):
        # reset to the begining of the demo
        env.reset(options={"demo": demos[i]})
        steps = len(demos[i]._observations)
        for step in range(steps - 1):
            joint_positions = demos[i]._observations[step].joint_positions
            gripper_joint_position = (
                demos[i]._observations[step].gripper_joint_positions[0]
            )
            action = np.concatenate(
                [joint_positions, np.array([gripper_joint_position])]
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"Step:{step} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}"
            )
            env.render()

if False:  # USING EE_POSE
    env = gym.make(
        "rlbench/close_drawer-state-v0",
        render_mode="human",
        action_type="ee_pose_relative",
        observation_type="state",
        max_episode_steps=200,
    )

    # create a demo
    print("Creating a demo")
    demos = env.unwrapped.get_demos(1, live_demos=True)

    print("Following the actions of the demo")
    # run the environment witht the same action as the demo
    for i in range(len(demos)):
        # reset to the begining of the demo
        env.reset(options={"demo": demos[i]})
        steps = len(demos[i]._observations)
        for step in range(steps - 1):
            gripper_pose_current = demos[i]._observations[step].gripper_pose
            gripper_pose_next = demos[i]._observations[step + 1].gripper_pose
            gripper_pose_relative = get_relative_pose(
                gripper_pose_current, gripper_pose_next
            )
            gripper_open = demos[i]._observations[step].gripper_open
            action = np.concatenate([gripper_pose_relative, np.array([gripper_open])])
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"Step:{step} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}"
            )
            env.render()

if True:
    env = gym.make(
        "rlbench/close_drawer-state-v0",
        render_mode="human",
        action_type="ee_pose_relative",
        observation_type="state",
        max_episode_steps=200,
    )
    # now just run normal RL loop
    episodes = 5
    for i in range(episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        counter = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info["failed_step"]:
                print("Failed step")

            print(
                f"Step:{counter} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}"
            )
            env.render()  # Note: rendering increases step time.
            counter += 1

    print("Done")

    fps = benchmark_step(env, target_duration=10)
    print(f"FPS: {fps:.2f}")


env.close()
