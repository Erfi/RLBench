import importlib
import pickle
from os import listdir
from os.path import join, exists
from typing import List

import numpy as np
from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor
from gymnasium.spaces import Box
from pyquaternion import Quaternion

from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig


class InvalidTaskName(Exception):
    pass


def name_to_task_class(task_file: str):
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    try:
        mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled." % name
        ) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'." % (class_name, name)
        ) from e
    return task_class


def get_stored_demos(
    amount: int,
    image_paths: bool,
    dataset_root: str,
    variation_number: int,
    task_name: str,
    obs_config: ObservationConfig,
    random_selection: bool = True,
    from_episode_number: int = 0,
) -> List[Demo]:

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError(
            "Can't find the demos for %s at: %s" % (task_name, task_root)
        )

    # Sample an amount of examples for the variation of this task
    examples_path = join(
        task_root, VARIATIONS_FOLDER % variation_number, EPISODES_FOLDER
    )
    examples = listdir(examples_path)
    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            "You asked for %d examples, but only %d were available."
            % (amount, len(examples))
        )
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(examples)[
            from_episode_number : from_episode_number + amount
        ]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        with open(join(example_path, LOW_DIM_PICKLE), "rb") as f:
            obs = pickle.load(f)

        l_sh_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
        l_sh_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
        l_sh_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
        r_sh_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
        r_sh_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        r_sh_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
        oh_rgb_f = join(example_path, OVERHEAD_RGB_FOLDER)
        oh_depth_f = join(example_path, OVERHEAD_DEPTH_FOLDER)
        oh_mask_f = join(example_path, OVERHEAD_MASK_FOLDER)
        wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
        wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
        wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
        front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
        front_mask_f = join(example_path, FRONT_MASK_FOLDER)

        num_steps = len(obs)

        if not (
            num_steps
            == len(listdir(l_sh_rgb_f))
            == len(listdir(l_sh_depth_f))
            == len(listdir(r_sh_rgb_f))
            == len(listdir(r_sh_depth_f))
            == len(listdir(oh_rgb_f))
            == len(listdir(oh_depth_f))
            == len(listdir(wrist_rgb_f))
            == len(listdir(wrist_depth_f))
            == len(listdir(front_rgb_f))
            == len(listdir(front_depth_f))
        ):
            raise RuntimeError("Broken dataset assumption")

        for i in range(num_steps):
            si = IMAGE_FORMAT % i
            if obs_config.left_shoulder_camera.rgb:
                obs[i].left_shoulder_rgb = join(l_sh_rgb_f, si)
            if (
                obs_config.left_shoulder_camera.depth
                or obs_config.left_shoulder_camera.point_cloud
            ):
                obs[i].left_shoulder_depth = join(l_sh_depth_f, si)
            if obs_config.left_shoulder_camera.mask:
                obs[i].left_shoulder_mask = join(l_sh_mask_f, si)
            if obs_config.right_shoulder_camera.rgb:
                obs[i].right_shoulder_rgb = join(r_sh_rgb_f, si)
            if (
                obs_config.right_shoulder_camera.depth
                or obs_config.right_shoulder_camera.point_cloud
            ):
                obs[i].right_shoulder_depth = join(r_sh_depth_f, si)
            if obs_config.right_shoulder_camera.mask:
                obs[i].right_shoulder_mask = join(r_sh_mask_f, si)
            if obs_config.overhead_camera.rgb:
                obs[i].overhead_rgb = join(oh_rgb_f, si)
            if (
                obs_config.overhead_camera.depth
                or obs_config.overhead_camera.point_cloud
            ):
                obs[i].overhead_depth = join(oh_depth_f, si)
            if obs_config.overhead_camera.mask:
                obs[i].overhead_mask = join(oh_mask_f, si)
            if obs_config.wrist_camera.rgb:
                obs[i].wrist_rgb = join(wrist_rgb_f, si)
            if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                obs[i].wrist_depth = join(wrist_depth_f, si)
            if obs_config.wrist_camera.mask:
                obs[i].wrist_mask = join(wrist_mask_f, si)
            if obs_config.front_camera.rgb:
                obs[i].front_rgb = join(front_rgb_f, si)
            if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                obs[i].front_depth = join(front_depth_f, si)
            if obs_config.front_camera.mask:
                obs[i].front_mask = join(front_mask_f, si)

            # Remove low dim info if necessary
            if not obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not obs_config.joint_positions:
                obs[i].joint_positions = None
            if not obs_config.joint_forces:
                obs[i].joint_forces = None
            if not obs_config.gripper_open:
                obs[i].gripper_open = None
            if not obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        if not image_paths:
            for i in range(num_steps):
                if obs_config.left_shoulder_camera.rgb:
                    obs[i].left_shoulder_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_rgb),
                            obs_config.left_shoulder_camera.image_size,
                        )
                    )
                if obs_config.right_shoulder_camera.rgb:
                    obs[i].right_shoulder_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].right_shoulder_rgb),
                            obs_config.right_shoulder_camera.image_size,
                        )
                    )
                if obs_config.overhead_camera.rgb:
                    obs[i].overhead_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].overhead_rgb),
                            obs_config.overhead_camera.image_size,
                        )
                    )
                if obs_config.wrist_camera.rgb:
                    obs[i].wrist_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_rgb),
                            obs_config.wrist_camera.image_size,
                        )
                    )
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].front_rgb),
                            obs_config.front_camera.image_size,
                        )
                    )

                if (
                    obs_config.left_shoulder_camera.depth
                    or obs_config.left_shoulder_camera.point_cloud
                ):
                    l_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_depth),
                            obs_config.left_shoulder_camera.image_size,
                        ),
                        DEPTH_SCALE,
                    )
                    near = obs[i].misc["left_shoulder_camera_near"]
                    far = obs[i].misc["left_shoulder_camera_far"]
                    l_sh_depth_m = near + l_sh_depth * (far - near)
                    if obs_config.left_shoulder_camera.depth:
                        d = (
                            l_sh_depth_m
                            if obs_config.left_shoulder_camera.depth_in_meters
                            else l_sh_depth
                        )
                        obs[i].left_shoulder_depth = (
                            obs_config.left_shoulder_camera.depth_noise.apply(d)
                        )
                    else:
                        obs[i].left_shoulder_depth = None

                if (
                    obs_config.right_shoulder_camera.depth
                    or obs_config.right_shoulder_camera.point_cloud
                ):
                    r_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].right_shoulder_depth),
                            obs_config.right_shoulder_camera.image_size,
                        ),
                        DEPTH_SCALE,
                    )
                    near = obs[i].misc["right_shoulder_camera_near"]
                    far = obs[i].misc["right_shoulder_camera_far"]
                    r_sh_depth_m = near + r_sh_depth * (far - near)
                    if obs_config.right_shoulder_camera.depth:
                        d = (
                            r_sh_depth_m
                            if obs_config.right_shoulder_camera.depth_in_meters
                            else r_sh_depth
                        )
                        obs[i].right_shoulder_depth = (
                            obs_config.right_shoulder_camera.depth_noise.apply(d)
                        )
                    else:
                        obs[i].right_shoulder_depth = None

                if (
                    obs_config.overhead_camera.depth
                    or obs_config.overhead_camera.point_cloud
                ):
                    oh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].overhead_depth),
                            obs_config.overhead_camera.image_size,
                        ),
                        DEPTH_SCALE,
                    )
                    near = obs[i].misc["overhead_camera_near"]
                    far = obs[i].misc["overhead_camera_far"]
                    oh_depth_m = near + oh_depth * (far - near)
                    if obs_config.overhead_camera.depth:
                        d = (
                            oh_depth_m
                            if obs_config.overhead_camera.depth_in_meters
                            else oh_depth
                        )
                        obs[i].overhead_depth = (
                            obs_config.overhead_camera.depth_noise.apply(d)
                        )
                    else:
                        obs[i].overhead_depth = None

                if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                    wrist_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_depth),
                            obs_config.wrist_camera.image_size,
                        ),
                        DEPTH_SCALE,
                    )
                    near = obs[i].misc["wrist_camera_near"]
                    far = obs[i].misc["wrist_camera_far"]
                    wrist_depth_m = near + wrist_depth * (far - near)
                    if obs_config.wrist_camera.depth:
                        d = (
                            wrist_depth_m
                            if obs_config.wrist_camera.depth_in_meters
                            else wrist_depth
                        )
                        obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(
                            d
                        )
                    else:
                        obs[i].wrist_depth = None

                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    front_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].front_depth),
                            obs_config.front_camera.image_size,
                        ),
                        DEPTH_SCALE,
                    )
                    near = obs[i].misc["front_camera_near"]
                    far = obs[i].misc["front_camera_far"]
                    front_depth_m = near + front_depth * (far - near)
                    if obs_config.front_camera.depth:
                        d = (
                            front_depth_m
                            if obs_config.front_camera.depth_in_meters
                            else front_depth
                        )
                        obs[i].front_depth = obs_config.front_camera.depth_noise.apply(
                            d
                        )
                    else:
                        obs[i].front_depth = None

                if obs_config.left_shoulder_camera.point_cloud:
                    obs[i].left_shoulder_point_cloud = (
                        VisionSensor.pointcloud_from_depth_and_camera_params(
                            l_sh_depth_m,
                            obs[i].misc["left_shoulder_camera_extrinsics"],
                            obs[i].misc["left_shoulder_camera_intrinsics"],
                        )
                    )
                if obs_config.right_shoulder_camera.point_cloud:
                    obs[i].right_shoulder_point_cloud = (
                        VisionSensor.pointcloud_from_depth_and_camera_params(
                            r_sh_depth_m,
                            obs[i].misc["right_shoulder_camera_extrinsics"],
                            obs[i].misc["right_shoulder_camera_intrinsics"],
                        )
                    )
                if obs_config.overhead_camera.point_cloud:
                    obs[i].overhead_point_cloud = (
                        VisionSensor.pointcloud_from_depth_and_camera_params(
                            oh_depth_m,
                            obs[i].misc["overhead_camera_extrinsics"],
                            obs[i].misc["overhead_camera_intrinsics"],
                        )
                    )
                if obs_config.wrist_camera.point_cloud:
                    obs[i].wrist_point_cloud = (
                        VisionSensor.pointcloud_from_depth_and_camera_params(
                            wrist_depth_m,
                            obs[i].misc["wrist_camera_extrinsics"],
                            obs[i].misc["wrist_camera_intrinsics"],
                        )
                    )
                if obs_config.front_camera.point_cloud:
                    obs[i].front_point_cloud = (
                        VisionSensor.pointcloud_from_depth_and_camera_params(
                            front_depth_m,
                            obs[i].misc["front_camera_extrinsics"],
                            obs[i].misc["front_camera_intrinsics"],
                        )
                    )

                # Masks are stored as coded RGB images.
                # Here we transform them into 1 channel handles.
                if obs_config.left_shoulder_camera.mask:
                    obs[i].left_shoulder_mask = rgb_handles_to_mask(
                        np.array(
                            _resize_if_needed(
                                Image.open(obs[i].left_shoulder_mask),
                                obs_config.left_shoulder_camera.image_size,
                            )
                        )
                    )
                if obs_config.right_shoulder_camera.mask:
                    obs[i].right_shoulder_mask = rgb_handles_to_mask(
                        np.array(
                            _resize_if_needed(
                                Image.open(obs[i].right_shoulder_mask),
                                obs_config.right_shoulder_camera.image_size,
                            )
                        )
                    )
                if obs_config.overhead_camera.mask:
                    obs[i].overhead_mask = rgb_handles_to_mask(
                        np.array(
                            _resize_if_needed(
                                Image.open(obs[i].overhead_mask),
                                obs_config.overhead_camera.image_size,
                            )
                        )
                    )
                if obs_config.wrist_camera.mask:
                    obs[i].wrist_mask = rgb_handles_to_mask(
                        np.array(
                            _resize_if_needed(
                                Image.open(obs[i].wrist_mask),
                                obs_config.wrist_camera.image_size,
                            )
                        )
                    )
                if obs_config.front_camera.mask:
                    obs[i].front_mask = rgb_handles_to_mask(
                        np.array(
                            _resize_if_needed(
                                Image.open(obs[i].front_mask),
                                obs_config.front_camera.image_size,
                            )
                        )
                    )

        demos.append(obs)
    return demos


def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image


class GripperPoseBox(Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self):
        """
        The pose is the concatenation of the position and the quaternion and the gripper open/close.
        In total 8 values: [x, y, z, qx, qy, qz, qw, open/close]
        [qx, qy, qz, qw] is the unit quaternion and should have the norm of 1.
        """
        raw_action = super().sample()  # [x,y,z, a,b,c, theta, open/close]
        pos = raw_action[:3]
        raw_quat = raw_action[3:-1]
        gripper_action = raw_action[-1]
        # we are using the unit quaternion: cos(theta/2) + sin(theta/2) * [a,b,c]
        abc = raw_quat[:-1] / np.linalg.norm(raw_quat[:-1])
        angle = raw_quat[-1] / 2  # angle in radians
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        quat = np.array(
            [abc[0] * sin_angle, abc[1] * sin_angle, abc[2] * sin_angle, cos_angle]
        )
        return np.concatenate([pos, quat, [gripper_action]])


def get_relative_pose(pose1, pose2):
    """
    Returns the relative pose from pose1 (current) to pose2 (next)
    Each pose is a 7D vector [x, y, z, qx, qy, qz, qw] with quaternion in the order of [x, y, z, w]
    """
    relative_position = get_relative_position(pose1[:3], pose2[:3])
    relative_quaternion = get_relative_quaternion(pose1[3:], pose2[3:])
    return np.concatenate([relative_position, relative_quaternion])


def get_relative_position(pos1, pos2):
    return pos2 - pos1


def get_relative_quaternion(quat1, quat2):
    """
    quat1 and quat2 are 4D vectors [x, y, z, w] with w being the scalar part
    returns the relative quaternion from quat1 to quat2 with the scalar part being positive
    """
    if quat1[-1] < 0:
        quat1 = -quat1
    if quat2[-1] < 0:
        quat2 = -quat2
    quat1 = Quaternion(quat1[-1], *quat1[:-1])
    quat2 = Quaternion(quat2[-1], *quat2[:-1])
    relative_quat = quat2 * quat1.inverse
    relative_quat = np.array(
        [relative_quat.x, relative_quat.y, relative_quat.z, relative_quat.w]
    )
    if relative_quat[-1] < 0:
        relative_quat = -relative_quat
    return relative_quat
