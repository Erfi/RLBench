from typing import Union, List, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

from rlbench.demo import Demo
from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.utils import GripperPoseBox


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
        observation_mode="state",
        render_mode: Union[None, str] = None,
        action_mode=None,
    ):
        self.task_class = task_class
        self.observation_mode = observation_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == "state":
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == "vision":
            obs_config.set_all(True)
        else:
            raise ValueError("Unrecognised observation_mode: %s." % observation_mode)
        self.obs_config = obs_config
        if action_mode is None:
            action_mode = JointPositionActionMode()
        self.action_mode = action_mode

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
        if self.observation_mode == "state":
            gym_obs = self._extract_obs_state(obs)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=gym_obs.shape, dtype=gym_obs.dtype
            )

        elif self.observation_mode == "vision":
            gym_obs = self._extract_obs_vision(obs)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=gym_obs.shape, dtype=gym_obs.dtype
            )
        else:
            raise ValueError("Unrecognised observation_mode: %s." % observation_mode)

        action_low, action_high = action_mode.action_bounds()
        self.action_space = GripperPoseBox(
            low=np.float32(action_low),
            high=np.float32(action_high),
            shape=self.rlbench_env.action_shape,
            dtype=np.float32,
        )

    def _extract_obs_all(self, rlbench_obs):
        gym_obs = {}
        for state_name in [
            "joint_velocities",
            "joint_positions",
            "joint_forces",
            "gripper_open",
            "gripper_pose",
            "gripper_joint_positions",
            "gripper_touch_forces",
            "task_low_dim_state",
        ]:
            state_data = getattr(rlbench_obs, state_name)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                gym_obs[state_name] = state_data

        if self.observation_mode == "vision":
            gym_obs.update(
                {
                    "left_shoulder_rgb": rlbench_obs.left_shoulder_rgb,
                    "left_shoulder_depth": rlbench_obs.left_shoulder_depth,
                    "right_shoulder_rgb": rlbench_obs.right_shoulder_rgb,
                    "right_shoulder_depth": rlbench_obs.right_shoulder_depth,
                    "wrist_rgb": rlbench_obs.wrist_rgb,
                    "wrist_depth": rlbench_obs.wrist_depth,
                    "front_rgb": rlbench_obs.front_rgb,
                    "front_depth": rlbench_obs.front_depth,
                }
            )
        return gym_obs

    def _extract_obs_state(self, rlbench_obs):
        """
        Only include gripper_pose, gipper_open and task_low_dim_state
        """
        gym_obs = []
        for state_name in ["gripper_pose", "gripper_open", "task_low_dim_state"]:
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
        # turn into channel first
        gym_obs = np.transpose(gym_obs, (2, 0, 1))
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

        # extract from the raw observation
        if self.observation_mode == "state":
            extracted_obs = self._extract_obs_state(obs)
        elif self.observation_mode == "vision":
            extracted_obs = self._extract_obs_vision(obs)

        return extracted_obs, {"text_descriptions": descriptions}

    def step(self, action):
        info = {"failed_step": False}
        try:
            obs, reward, terminated = self.rlbench_task_env.step(action)
            # extract from the raw observation
            if self.observation_mode == "state":
                extracted_obs = self._extract_obs_state(obs)
            elif self.observation_mode == "vision":
                extracted_obs = self._extract_obs_vision(obs)

            return extracted_obs, reward, terminated, False, info
        except Exception as e:
            info["failed_step"] = True
            dummy_next_state = self.observation_space.sample() * -1
            # HACK: since terminated is True, the next_state is not used
            return dummy_next_state, 0, True, False, info

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
