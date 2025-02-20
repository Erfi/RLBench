from typing import Union, List, Dict
import logging
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

from cv2 import resize
from cv2 import INTER_NEAREST

from rlbench.demo import Demo
from rlbench.action_modes.action_mode import (
    JointPositionAbsoluteActionMode,
    JointPositionRelativeActionMode,
    EEPlannerAbsoluteActionMode,
    EEPlannerRelativeActionMode,
    EEIKRelativeActionMode,
    JointVelocityAbsoluteActionMode,
)

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.utils import GripperPoseBox


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def convert_dtype_to_float32_if_float(dtype):
    if issubclass(dtype.type, np.floating):
        return np.float32
    return dtype


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        task_class,
        observation_type="state",
        action_type="ee_pose_absolute",
        render_mode: Union[None, str] = None,
    ):
        self.task_class = task_class
        self.observation_type = observation_type
        self.action_type = action_type
        self.obs_history = deque(maxlen=2)  # used for getting relative actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        logging.info(
            f"Creating RLBenchEnv: observation_type={observation_type}, action_type={action_type}"
        )

        self.obs_config = ObservationConfig()
        if self.observation_type == "state":
            self.obs_config.set_all_high_dim(False)
            self.obs_config.set_all_low_dim(True)
        elif self.observation_type == "vision":
            self.obs_config.set_all(True)
        else:
            raise ValueError(
                "Unrecognised observation_type: %s." % self.observation_type
            )

        if self.action_type == "ee_pose_absolute":
            self.action_mode = EEPlannerAbsoluteActionMode()
        elif self.action_type == "ee_pose_relative":
            # self.action_mode = EEPlannerRelativeActionMode()
            self.action_mode = EEIKRelativeActionMode()
        elif self.action_type == "joint_position_absolute":
            self.action_mode = JointPositionAbsoluteActionMode()
        elif self.action_type == "joint_position_relative":
            self.action_mode = JointPositionRelativeActionMode()
        elif self.action_type == "joint_velocity_absolute":
            self.action_mode = JointVelocityAbsoluteActionMode()

        else:
            raise ValueError("Unrecognised action type: %s." % self.action_type)

        self.rlbench_env = Environment(
            action_mode=self.action_mode,
            obs_config=self.obs_config,
            headless=True,
        )
        self.rlbench_env.launch()
        self.rlbench_task_env = self.rlbench_env.get_task(self.task_class)
        if render_mode is not None:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self.gym_cam = VisionSensor.create([640, 360])
            self.gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self.gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self.gym_cam.set_render_mode(RenderMode.OPENGL3)

        # --- setup observation and action space ---
        _, obs = self.rlbench_task_env.reset()
        gym_obs = self._extract_obs(obs)

        if self.observation_type == "state":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=gym_obs.shape, dtype=gym_obs.dtype
            )

        elif self.observation_type == "vision":
            self.observation_space = spaces.Box(
                low=0, high=1.0, shape=gym_obs.shape, dtype=gym_obs.dtype
            )
        else:
            raise ValueError("Unrecognised observation_type: %s." % observation_type)

        action_low, action_high = self.action_mode.action_bounds()
        self.action_space = spaces.Box(
            low=np.float32(action_low),
            high=np.float32(action_high),
            shape=self.rlbench_env.action_shape,
            dtype=np.float32,
        )

        self.obs_history.append(gym_obs)

    def _extract_obs(self, rlbench_obs):
        if self.observation_type == "state":
            return self._extract_obs_state(rlbench_obs)
        elif self.observation_type == "vision":
            return self._extract_obs_vision(rlbench_obs)
        else:
            raise ValueError(
                "Unrecognised observation_type: %s." % self.observation_type
            )

    def _extract_obs_state(self, rlbench_obs):
        """
        Only include gripper_pose, gipper_open and task_low_dim_state
        """
        gym_obs = []
        for state_name in [
            "gripper_pose",
            "gripper_open",
            "joint_positions",
            "joint_velocities",
            "task_low_dim_state",
        ]:
            state_data = getattr(rlbench_obs, state_name)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                gym_obs.append(state_data)
        # concatenate all into a sinle array
        gym_obs = np.concatenate(gym_obs)
        return gym_obs

    def _extract_obs_vision(self, rlbench_obs):
        gym_obs = {
            "left_shoulder_rgb": rlbench_obs.left_shoulder_rgb,
            "left_shoulder_depth": rlbench_obs.left_shoulder_depth[..., np.newaxis],
            "right_shoulder_rgb": rlbench_obs.right_shoulder_rgb,
            "right_shoulder_depth": rlbench_obs.right_shoulder_depth[..., np.newaxis],
            "wrist_rgb": rlbench_obs.wrist_rgb,
            "wrist_depth": rlbench_obs.wrist_depth[..., np.newaxis],
            "front_rgb": rlbench_obs.front_rgb,
            "front_depth": rlbench_obs.front_depth[..., np.newaxis],
        }
        # concatenate all into a single array at the last dimension
        gym_obs = np.concatenate(list(gym_obs.values()), axis=-1)
        # resize to 64x64
        gym_obs = resize(gym_obs, (64, 64), interpolation=INTER_NEAREST)
        # turn into channel first
        gym_obs = np.transpose(gym_obs, (2, 0, 1))
        # scale to [0, 1]
        gym_obs = gym_obs / 255.0
        # convert to float32
        gym_obs = np.float32(gym_obs)
        return gym_obs

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self.gym_cam.capture_rgb()
            frame = np.clip((frame * 255.0).astype(np.uint8), 0, 255)
            return frame

    def reset(self, seed: int = None, options: Dict = None):
        super().reset(seed=seed)
        # TODO: Remove this and use seed from super()
        np.random.seed(seed=seed)
        demo = None
        if options is not None:
            # TODO: Write test for this
            demo = options.get("demo", None)

        if demo is None:
            descriptions, obs = self.rlbench_task_env.reset()
        else:
            descriptions, obs = self.rlbench_task_env.reset_to_demo(demo=demo)

        # reset obs_history
        self.obs_history = deque(maxlen=2)
        self.obs_history.append(obs)
        return self._extract_obs(obs), {"text_descriptions": descriptions}

    def step(self, action):
        info = {"failed_step": False}
        try:
            action = action.flatten()
            obs, reward, terminated = self.rlbench_task_env.step(action)
            self.obs_history.append(obs)
            return self._extract_obs(obs), reward, terminated, False, info
        except Exception as e:  # do nothing
            logging.info(f"Failed step: {e}")
            info["failed_step"] = True
            dummy_next_state = self._extract_obs(
                self.obs_history[-1]
            )  # Don't change the state
            return (dummy_next_state, 0.0, True, False, info)

    def close(self) -> None:
        self.rlbench_env.shutdown()

    def get_demos(
        self,
        amount: int,
        live_demos: bool = False,
        image_paths: bool = False,
        callable_each_step=None,
        max_attempts=10,
        random_selection=True,
        from_episode_number=0,
    ) -> List[Demo]:
        return self.rlbench_task_env.get_demos(
            amount,
            live_demos,
            image_paths,
            callable_each_step,
            max_attempts,
            random_selection,
            from_episode_number,
        )
