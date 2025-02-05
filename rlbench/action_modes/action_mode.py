from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperActionMode, GripperJointPosition, Discrete
from rlbench.backend.scene import Scene


class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
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
        raise NotImplementedError('You must define your own action bounds.')


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
            self.gripper_action_mode.action_shape(scene))


# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.

class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True))

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
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.1] + [0.0]), np.array(7 * [0.1] + [0.04])

class EEPlannerActionMode(ActionMode):
    """
    An action mode that allows the end effector to be moved to a target pose
    using the planner. The gripper is controlled using a discrete action space.
    """
    def __init__(self):
        super(EEPlannerActionMode, self).__init__(
            EndEffectorPoseViaPlanning(absolute_mode=True), Discrete())

    def action(self, scene, action):
        # create a valid quternion (qx, qy, qz, qw) from the action
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        pos = arm_action[:3]
        raw_quat = arm_action[3:]
        quat = raw_quat / np.linalg.norm(raw_quat)

        # apply arm action
        arm_action = np.concatenate([pos, quat])
        self.arm_action_mode.action(scene, arm_action)

        # apply gripper action
        ee_action = np.array(action[arm_act_size:])
        self.gripper_action_mode.action(scene, ee_action)


    def action_shape(self, scene):
        # 3 position + 4 quaternion (qx, qy, qz, qw) + 1 gripper
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(self.gripper_action_mode.action_shape(scene))
    
    def action_bounds(self):
        """Returns the min and max of the action mode."""
        low = np.array([-0.32, -0.45, 0.75] + 4*[-1.0] + [0.0])
        high = np.array([0.32, 0.45, 1.75] + 4*[1.0] + [1.0])
        return low, high