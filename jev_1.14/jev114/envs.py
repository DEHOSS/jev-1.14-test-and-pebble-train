"""Gym-like adapters, MetaWorld channel schema, and trajectory text for Jev."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


TASKS = {
    "walker": "Control a planar bipedal walker to move forward steadily without falling.",
    "cheetah": "Control a planar cheetah to run forward as fast as possible.",
    "sweep": "Sweep the object into the target area with the robot end effector.",
    "push-button": "Press the button.",
}

METAWORLD_TASKS = {
    "sweep": "sweep-into",
    "push-button": "button-press",
}

ENV_CHOICES = ("push-button", "walker", "cheetah", "sweep")


@dataclass(frozen=True)
class EnvSpec:
    name: str
    task: str
    obs_names: tuple[str, ...]
    action_names: tuple[str, ...]
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int


def is_metaworld(name: str) -> bool:
    return name in METAWORLD_TASKS


def _flat_dm(observation: dict) -> np.ndarray:
    return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in observation.values()])


class DMCEnv:
    def __init__(self, name: str, seed: int):
        from dm_control import suite

        domain, task = {"walker": ("walker", "walk"), "cheetah": ("cheetah", "run")}[name]
        self.env = suite.load(domain, task, task_kwargs={"random": seed})
        obs_names = []
        for key, value in self.env.observation_spec().items():
            size = int(np.prod(value.shape))
            obs_names.extend([key] if size == 1 else [f"{key}_{i}" for i in range(size)])
        act = self.env.action_spec()
        low = np.broadcast_to(act.minimum, act.shape).astype(np.float32).copy()
        high = np.broadcast_to(act.maximum, act.shape).astype(np.float32).copy()
        self.spec = EnvSpec(name, TASKS[name], tuple(obs_names),
                            tuple(f"action_{i}" for i in range(low.size)), low, high,
                            1000)

    def reset(self, seed=None):
        # dm_control seeds the task when it is constructed.
        return _flat_dm(self.env.reset().observation), {}

    def step(self, action):
        ts = self.env.step(np.asarray(action, dtype=np.float32))
        return _flat_dm(ts.observation), float(ts.reward or 0.0), False, bool(ts.last()), {}

    def get_sim_state(self):
        return np.array(self.env.physics.get_state(), copy=True)

    def set_sim_state(self, state):
        self.env.physics.set_state(np.asarray(state))
        self.env.physics.forward()

    def render_rgb(self, width: int | None = None, height: int | None = None,
                   camera: str = "corner2") -> np.ndarray:
        del camera
        w = 640 if width is None else int(width)
        h = 480 if height is None else int(height)
        return np.ascontiguousarray(self.env.physics.render(height=h, width=w))

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


class MetaWorldEnv:
    def __init__(self, name: str, seed: int, version: int = 3, max_episode_steps: int = 1000):
        import metaworld

        env_name = f"{METAWORLD_TASKS[name]}-v{version}"
        benchmark = metaworld.MT1(env_name, seed=seed)
        self.env = benchmark.train_classes[env_name]()
        self.tasks = list(benchmark.train_tasks)
        self.rng = np.random.default_rng(seed)
        self.action_space = self.env.action_space
        self.env.max_path_length = max_episode_steps
        obs_dim = int(np.prod(self.env.observation_space.shape))
        names = list(metaworld_observation_names())
        if len(names) < obs_dim:
            names.extend(f"obs_{i}" for i in range(len(names), obs_dim))
        names = names[:obs_dim]
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self.spec = EnvSpec(name, TASKS[name], tuple(names),
                            ("action_dx", "action_dy", "action_dz", "action_gripper"),
                            low, high, max_episode_steps)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.env.set_task(self.tasks[int(self.rng.integers(len(self.tasks)))])
        result = self.env.reset()
        if isinstance(result, tuple):
            return np.asarray(result[0], dtype=np.float32), result[1]
        return np.asarray(result, dtype=np.float32), {}

    def step(self, action):
        result = self.env.step(np.asarray(action, dtype=np.float32))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def get_sim_state(self):
        state = self.env.get_env_state()
        return tuple(np.array(part, copy=True) for part in state)

    def set_sim_state(self, state):
        self.env.set_env_state(tuple(np.array(part, copy=True) for part in state))

    def render_rgb(self, width: int | None = None, height: int | None = None,
                   camera: str = "corner2") -> np.ndarray:
        """Offscreen RGB frame. Callers should set ``MUJOCO_GL=egl`` on headless machines.

        Default camera is MetaWorld ``corner2`` (elevated 3/4 table view), not the
        tracking camera gymnasium uses when no camera is set.
        """
        import mujoco

        self.env.render_mode = "rgb_array"
        self.env.camera_name = camera
        self.env.camera_id = None
        renderer = getattr(self.env, "mujoco_renderer", None)
        if renderer is not None:
            if width and height:
                renderer.width = int(width)
                renderer.height = int(height)
            cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                raise ValueError(f"Unknown MetaWorld camera {camera!r}")
            renderer.camera_id = cam_id
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("MetaWorld render returned None; set MUJOCO_GL=egl")
        # EGL offscreen is bottom-up. Gymnasium also flips, but that pair still
        # leaves MetaWorld videos inverted on this stack, so flip once more.
        return np.ascontiguousarray(np.asarray(frame)[::-1])

    def close(self):
        self.env.close()


def make_env(name: str, seed: int, sweep_version: int = 3):
    if is_metaworld(name):
        return MetaWorldEnv(name, seed, sweep_version, max_episode_steps=1000)
    return DMCEnv(name, seed)


@dataclass(frozen=True)
class ChannelSpec:
    index: int
    name: str
    meaning: str
    units: str = "unitless"
    value_range: Optional[str] = None
    temporal_role: str = "current"

    def to_state_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class TrajectorySchema:
    environment_family: str
    observation_channels: tuple[ChannelSpec, ...]
    action_channels: tuple[ChannelSpec, ...]
    conventions: tuple[str, ...]

    def to_state_dict(self) -> dict:
        return {
            "environment_family": self.environment_family,
            "observation_channels": [channel.to_state_dict() for channel in self.observation_channels],
            "action_channels": [channel.to_state_dict() for channel in self.action_channels],
            "conventions": list(self.conventions),
        }


def metaworld_v2_schema() -> TrajectorySchema:
    """Full 39-D MetaWorld v2 observation + 4-D action schema used in prior Jev experiments."""
    obs: list[ChannelSpec] = []

    def add_state_block(offset: int, temporal_role: str) -> None:
        prefix = "" if temporal_role == "current" else "previous_"
        for j, axis in enumerate(("x", "y", "z")):
            obs.append(ChannelSpec(
                offset + j, f"{prefix}tcp_{axis}",
                f"{temporal_role.capitalize()} Cartesian {axis.upper()} "
                "coordinate of the robot tool-center point/end-effector.",
                units="meters", temporal_role=temporal_role,
            ))
        obs.append(ChannelSpec(
            offset + 3, f"{prefix}gripper_open",
            f"{temporal_role.capitalize()} gripper opening state. "
            "Larger values mean more open and smaller values mean more closed.",
            units="normalized simulator value",
            value_range="approximately [-1, 1] in MetaWorld",
            temporal_role=temporal_role,
        ))
        for j, axis in enumerate(("x", "y", "z")):
            obs.append(ChannelSpec(
                offset + 4 + j, f"{prefix}object1_{axis}",
                f"{temporal_role.capitalize()} Cartesian {axis.upper()} "
                "coordinate of the primary manipulated object.",
                units="meters", temporal_role=temporal_role,
            ))
        for j, q in enumerate(("qx", "qy", "qz", "qw")):
            obs.append(ChannelSpec(
                offset + 7 + j, f"{prefix}object1_{q}",
                f"{temporal_role.capitalize()} quaternion component {q} "
                "describing the orientation of the primary object.",
                units="unit quaternion component", value_range="[-1, 1]",
                temporal_role=temporal_role,
            ))
        for j, axis in enumerate(("x", "y", "z")):
            obs.append(ChannelSpec(
                offset + 11 + j, f"{prefix}object2_{axis}",
                f"{temporal_role.capitalize()} Cartesian {axis.upper()} "
                "coordinate of the secondary object. For single-object tasks "
                "this slot may be an unused placeholder.",
                units="meters", temporal_role=temporal_role,
            ))
        for j, q in enumerate(("qx", "qy", "qz", "qw")):
            obs.append(ChannelSpec(
                offset + 14 + j, f"{prefix}object2_{q}",
                f"{temporal_role.capitalize()} quaternion component {q} "
                "describing the secondary object's orientation. For "
                "single-object tasks this slot may be an unused placeholder.",
                units="unit quaternion component", value_range="[-1, 1]",
                temporal_role=temporal_role,
            ))

    add_state_block(0, "current")
    add_state_block(18, "previous")
    for j, axis in enumerate(("x", "y", "z")):
        obs.append(ChannelSpec(
            36 + j, f"goal_{axis}",
            f"Cartesian {axis.upper()} coordinate of the task goal/target. "
            "In a goal-hidden/partially-observable MetaWorld variant this "
            "field may be zeroed by the environment.",
            units="meters", temporal_role="goal",
        ))
    actions = (
        ChannelSpec(0, "action_dx",
                    "Command controlling end-effector displacement/motion along X.",
                    units="normalized control command", value_range="[-1, 1]",
                    temporal_role="action"),
        ChannelSpec(1, "action_dy",
                    "Command controlling end-effector displacement/motion along Y.",
                    units="normalized control command", value_range="[-1, 1]",
                    temporal_role="action"),
        ChannelSpec(2, "action_dz",
                    "Command controlling end-effector displacement/motion along Z.",
                    units="normalized control command", value_range="[-1, 1]",
                    temporal_role="action"),
        ChannelSpec(3, "action_gripper",
                    "Gripper control command. Its sign/magnitude controls closing versus "
                    "opening according to the MetaWorld actuator convention.",
                    units="normalized control command", value_range="[-1, 1]",
                    temporal_role="action"),
    )
    conventions = (
        "The observation is a 39-D MetaWorld v2 vector.",
        "Dimensions 0-17 describe the current state.",
        "Dimensions 18-35 repeat the same layout for the previous state.",
        "Dimensions 36-38 contain the goal position.",
        "Object #2 fields can be placeholders in tasks that manipulate only one object.",
        "Quaternion channels jointly represent orientation; do not interpret one "
        "quaternion component alone as an angle.",
        "Judge task success from temporal relationships among TCP, objects, goal, "
        "gripper state, and actions rather than from one isolated scalar.",
    )
    return TrajectorySchema("MetaWorld v2", tuple(obs), actions, conventions)


def metaworld_observation_names() -> tuple[str, ...]:
    return tuple(channel.name for channel in metaworld_v2_schema().observation_channels)


def feature_schema(spec: EnvSpec) -> dict:
    """Channel meanings sent before numeric trajectories, matching prior Jev payloads."""
    if is_metaworld(spec.name):
        return metaworld_v2_schema().to_state_dict()
    observations = tuple(
        ChannelSpec(i, name, f"Named observation channel `{name}` from the simulator.",
                    temporal_role="current")
        for i, name in enumerate(spec.obs_names)
    )
    actions = tuple(
        ChannelSpec(i, name, f"Named action channel `{name}`. Commands are normalized to [-1, 1].",
                    units="normalized control command", value_range="[-1, 1]",
                    temporal_role="action")
        for i, name in enumerate(spec.action_names)
    )
    return TrajectorySchema(
        "dm_control", observations, actions,
        ("Each named channel is a time-ordered sequence over the segment.",
         "Judge from temporal evolution of the state and actions, not isolated scalars."),
    ).to_state_dict()


def json_float(value, precision: int = 4) -> float:
    """Round to ``precision`` decimal places using Python ``round``.

    ``np.round`` on float32 keeps a binary value whose ``json.dumps`` form is
    ``0.11500000208616257``. Python ``round`` yields a float that serializes as
    ``0.115``.
    """
    if precision < 0:
        raise ValueError("precision cannot be negative")
    return round(float(value), int(precision))


def round_json_numbers(obj, precision: int = 4):
    """Recursively replace floats so JSON does not emit float32 tails."""
    if isinstance(obj, dict):
        return {key: round_json_numbers(value, precision) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_json_numbers(value, precision) for value in obj]
    if isinstance(obj, (float, np.floating)):
        return json_float(obj, precision)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return round_json_numbers(obj.tolist(), precision)
    return obj


def _aligned_arrays(observations: np.ndarray, actions: np.ndarray, spec: EnvSpec):
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError("observations and actions must be aligned 2-D arrays")
    if observations.shape[1] != len(spec.obs_names) or actions.shape[1] != len(spec.action_names):
        raise ValueError("trajectory shape does not match environment specification")
    return observations, actions


def trajectory_state(observations: np.ndarray, actions: np.ndarray, spec: EnvSpec,
                     precision: int = 4) -> dict:
    """Named channel sequences for OpenRouter state; never includes ground-truth rewards."""
    observations, actions = _aligned_arrays(observations, actions, spec)
    return {
        "num_steps": int(len(actions)),
        "observation_sequences": {
            name: [json_float(value, precision) for value in observations[:, i]]
            for i, name in enumerate(spec.obs_names)
        },
        "action_sequences": {
            name: [json_float(value, precision) for value in actions[:, i]]
            for i, name in enumerate(spec.action_names)
        },
    }


def trajectory_text(observations: np.ndarray, actions: np.ndarray, spec: EnvSpec,
                    precision: int = 3) -> str:
    """Channel sequences for logs; never includes ground-truth rewards."""
    state = trajectory_state(observations, actions, spec, precision=precision)
    lines = [f"num_steps: {state['num_steps']}"]
    for name, values in state["observation_sequences"].items():
        lines.append(f"{name}: [" + ", ".join(f"{v:.{precision}f}" for v in values) + "]")
    for name, values in state["action_sequences"].items():
        lines.append(f"{name}: [" + ", ".join(f"{v:.{precision}f}" for v in values) + "]")
    return "\n".join(lines)
