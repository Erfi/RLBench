from abc import abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    JointPosition,
    JointVelocity,
    EndEffectorPoseViaPlanning,
    EndEffectorPoseViaIK,
)
from rlbench.action_modes.gripper_action_modes import (
    GripperActionMode,
    GripperJointPosition,
    Discrete,
)
from rlbench.backend.scene import Scene


class ActionMode(object):

    def __init__(
        self, arm_action_mode: "ArmActionMode", gripper_action_mode: "GripperActionMode"
    ):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise NotImplementedError("You must define your own action bounds.")


class MoveArmThenGripper(ActionMode):
    """A customizable action mode.

    The arm action is first applied, followed by the gripper action.
    """

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)
        )


# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.


class JointPositionAbsoluteActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionAbsoluteActionMode, self).__init__(
            JointPosition(True), GripperJointPosition(True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)
        )

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-np.pi] + [0.0]), np.array(7 * [np.pi] + [0.04 + 1e-4])


class JointPositionRelativeActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionRelativeActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)
        )

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.01] + [0.0]), np.array(7 * [0.01] + [0.04 + 1e-4])


class JointVelocityAbsoluteActionMode(ActionMode):
    def __init__(self):
        super(JointVelocityAbsoluteActionMode, self).__init__(
            JointVelocity(), GripperJointPosition(True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)
        )

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        # TODO: check the bounds (from Demo)
        return np.array(7 * [-0.8] + [0.0]), np.array(7 * [0.8] + [0.04 + 1e-4])


class EEPlannerAbsoluteActionMode(ActionMode):
    """
    An action mode that allows the end effector to be moved to a target pose
    using the planner. The gripper is controlled using a discrete action space.
    """

    def __init__(self):
        super(EEPlannerAbsoluteActionMode, self).__init__(
            EndEffectorPoseViaPlanning(absolute_mode=True), Discrete()
        )

    def action(self, scene, action):
        # --- using quaternion ---
        arm_act_size = 7  # 3 position + 4 quaternion
        arm_action = np.array(action[:arm_act_size])
        # make unit_quaternion
        arm_action[3:] = arm_action[3:] / np.linalg.norm(arm_action[3:])
        if arm_action[3] < 0:
            arm_action[3:] = -arm_action[3:]
        gripper_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, gripper_action)

        # --- converting from Euler angles to quaternions ---
        # # create a valid quternion (qx, qy, qz, qw) from Euler angles (theta_x, theta_y, theta_z)
        # arm_act_size = 6  # 3 position + 3 Euler angles
        # arm_action = np.array(action[:arm_act_size])
        # pos = arm_action[:3]
        # euler = arm_action[3:]
        # rot = Rotation.from_euler("xyz", euler, degrees=True)
        # quat = rot.as_quat(canonical=True, scalar_first=False)

        # # apply arm action
        # arm_action = np.concatenate([pos, quat])
        # self.arm_action_mode.action(scene, arm_action)

        # # apply gripper action
        # ee_action = np.array(action[arm_act_size:])
        # self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene):
        # 3 position + quoternion (qx, qy, qz, qw) + gripper
        return 8

        # 3 position + 3 Euler angles (theta_x, theta_y, theta_z) + 1 gripper
        # return 7

    def action_bounds(self):
        """
        Returns the min and max of the action mode.
        [x,y,z,theta_x, theta_y, theta_z, gripper_open]
        all angles are in degrees
        """
        # --- using quaternion ---
        low = np.array([-0.32, -0.45, 0.75] + 3 * [-1.0] + [0.0] + [0.0])
        high = np.array([0.32, 0.45, 1.7] + 3 * [1.0] + [1.0] + [1.0])
        # --- using Euler angles ---
        # low = np.array([-0.32, -0.45, 0.75] + 3 * [-180] + [0.0])
        # high = np.array([0.32, 0.45, 1.7] + 3 * [180] + [1.0])
        return low, high


class EEPlannerRelativeActionMode(ActionMode):
    """
    An action mode that allows the end effector to be moved to a target pose
    using the planner. The gripper is controlled using a discrete action space.
    """

    def __init__(self):
        super(EEPlannerRelativeActionMode, self).__init__(
            EndEffectorPoseViaPlanning(absolute_mode=False), Discrete()
        )

    def action(self, scene, action):
        # create a valid quternion (qx, qy, qz, qw) from Euler angles (theta_x, theta_y, theta_z)
        arm_act_size = 6  # 3 position + 3 Euler angles
        arm_action = np.array(action[:arm_act_size])
        pos = arm_action[:3]
        euler = arm_action[3:]
        rot = Rotation.from_euler("xyz", euler, degrees=True)
        quat = rot.as_quat(canonical=True, scalar_first=False)

        # apply arm action
        arm_action = np.concatenate([pos, quat])
        self.arm_action_mode.action(scene, arm_action)

        # apply gripper action
        ee_action = np.array(action[arm_act_size:])
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene):
        # 3 position + 3 Euler angles (theta_x, theta_y, theta_z) + 1 gripper
        return 7

    def action_bounds(self):
        """
        Returns the min and max of the action mode.
        [x,y,z,theta_x, theta_y, theta_z, gripper_open]
        all angles are in degrees
        """

        low = np.array([-0.05, -0.05, -0.05] + 3 * [-1.0] + [0.0])
        high = np.array([0.05, 0.05, 0.05] + 3 * [1.0] + [1.0])
        return low, high


class EEIKRelativeActionMode(ActionMode):
    """
    An action mode that allows the end effector to be moved to a target pose
    using the planner. The gripper is controlled using a discrete action space.
    """

    def __init__(self):
        super(EEIKRelativeActionMode, self).__init__(
            EndEffectorPoseViaIK(absolute_mode=False), Discrete()
        )

    def action(self, scene, action):
        # create a valid quternion (qx, qy, qz, qw) from Euler angles (theta_x, theta_y, theta_z)
        arm_act_size = 6  # 3 position + 3 Euler angles
        arm_action = np.array(action[:arm_act_size])
        pos = arm_action[:3]
        euler = arm_action[3:]
        rot = Rotation.from_euler("xyz", euler, degrees=True)
        quat = rot.as_quat(canonical=True, scalar_first=False)

        # apply arm action
        arm_action = np.concatenate([pos, quat])
        self.arm_action_mode.action(scene, arm_action)

        # apply gripper action
        ee_action = np.array(action[arm_act_size:])
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene):
        # 3 position + 3 Euler angles (theta_x, theta_y, theta_z) + 1 gripper
        return 7

    def action_bounds(self):
        """
        Returns the min and max of the action mode.
        [x,y,z,theta_x, theta_y, theta_z, gripper_open]
        all angles are in degrees
        """

        low = np.array([-0.01, -0.01, -0.01] + 3 * [-1] + [0.0])
        high = np.array([0.01, 0.01, 0.01] + 3 * [1] + [1.0])
        return low, high


class EEIKAbsoluteActionMode(ActionMode):
    """
    An action mode that allows the end effector to be moved to a target pose
    using the planner. The gripper is controlled using a discrete action space.
    """

    def __init__(self):
        super(EEIKAbsoluteActionMode, self).__init__(
            EndEffectorPoseViaIK(absolute_mode=True), Discrete()
        )

    def action(self, scene, action):
        # create a valid quternion (qx, qy, qz, qw) from Euler angles (theta_x, theta_y, theta_z)
        arm_act_size = 6  # 3 position + 3 Euler angles
        arm_action = np.array(action[:arm_act_size])
        pos = arm_action[:3]
        euler = arm_action[3:]
        rot = Rotation.from_euler("xyz", euler, degrees=True)
        quat = rot.as_quat(canonical=True, scalar_first=False)

        # apply arm action
        arm_action = np.concatenate([pos, quat])
        self.arm_action_mode.action(scene, arm_action)

        # apply gripper action
        ee_action = np.array(action[arm_act_size:])
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene):
        # 3 position + 3 Euler angles (theta_x, theta_y, theta_z) + 1 gripper
        return 7

    def action_bounds(self):
        """
        Returns the min and max of the action mode.
        [x,y,z,theta_x, theta_y, theta_z, gripper_open]
        all angles are in degrees
        """

        low = np.array([-0.01, -0.01, -0.01] + 3 * [-1] + [0.0])
        high = np.array([0.01, 0.01, 0.01] + 3 * [1] + [1.0])
        return low, high
