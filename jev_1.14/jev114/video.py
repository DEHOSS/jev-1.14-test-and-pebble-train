"""Write MP4s of preference clips by replaying stored simulator states."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import traceback

import numpy as np

from .envs import is_metaworld
from .learning import Segment

# MetaWorld's first table-corner camera is named ``corner`` (not corner1).
METAWORLD_PAIR_CAMERAS = ("corner2", "corner")


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be (T, H, W, 3)")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
               f"{width}x{height}", "-framerate", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin, stderr = process.stdin, process.stderr
    try:
        if stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        for frame in frames:
            stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    except BrokenPipeError as exc:
        raise RuntimeError("ffmpeg closed stdin before the clip was written") from exc
    finally:
        if stdin is not None:
            stdin.close()
    errors = stderr.read().decode(errors="replace") if stderr is not None else ""
    status = process.wait()
    if stderr is not None:
        stderr.close()
    if status != 0:
        raise RuntimeError(f"ffmpeg failed: {errors.strip()}")


def render_segment_frames(env, segment: Segment, width: int, height: int,
                          camera: str = "corner2") -> np.ndarray:
    if not segment.states:
        raise ValueError("segment has no simulator states to render")
    frames = []
    for state in segment.states:
        env.set_sim_state(state)
        frames.append(env.render_rgb(width, height, camera=camera))
    return np.stack(frames)


class PairVideoSaver:
    """Save the first N accepted Jev pairs as clip-A / clip-B MP4s at each query pause."""

    def __init__(self, env, directory: Path, n_pairs: int,
                 width: int = 480, height: int = 480, fps: int = 8,
                 cameras: tuple[str, ...] | None = None):
        if n_pairs < 0:
            raise ValueError("n_pairs cannot be negative")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("pair-video width and height must be positive even numbers")
        if fps <= 0:
            raise ValueError("pair-video fps must be positive")
        self.env = env
        self.directory = directory
        self.n_per_query = n_pairs
        self.width = width
        self.height = height
        self.fps = fps
        if cameras is None:
            spec = getattr(env, "spec", None)
            name = getattr(spec, "name", None)
            cameras = METAWORLD_PAIR_CAMERAS if name and is_metaworld(name) else ("dmc",)
        self.cameras = cameras
        self.saved_this_query = 0
        self.total_saved = 0
        self.failures = 0
        if self.n_per_query:
            self.directory.mkdir(parents=True, exist_ok=True)

    def begin_query(self) -> None:
        self.saved_this_query = 0

    def maybe_save(self, clip_a: Segment, clip_b: Segment, env_steps: int,
                   feedback_step: int, record: dict) -> list[str] | None:
        if self.n_per_query <= 0 or self.saved_this_query >= self.n_per_query:
            return None
        try:
            paths = self._save(clip_a, clip_b, env_steps, feedback_step, record)
        except Exception as exc:
            self.failures += 1
            print(f"step={env_steps}: pair video save failed "
                  f"({type(exc).__name__}: {exc}); training continues", flush=True)
            traceback.print_exc()
            if self.failures >= 3:
                print(f"step={env_steps}: disabling further pair videos after "
                      f"{self.failures} failures", flush=True)
                self.n_per_query = 0
            return None
        self.saved_this_query += 1
        self.total_saved += 1
        self.failures = 0
        return paths

    def _save(self, clip_a: Segment, clip_b: Segment, env_steps: int,
              feedback_step: int, record: dict) -> list[str]:
        if not clip_a.states or not clip_b.states:
            raise RuntimeError("clips are missing simulator states")
        index = self.saved_this_query + 1
        stem = (f"step{env_steps}_fb{feedback_step:04d}_pair{index:02d}")
        videos = {"A": {}, "B": {}}
        paths = []
        errors = []
        for label, clip in (("A", clip_a), ("B", clip_b)):
            for camera in self.cameras:
                path = self.directory / f"{stem}_{label}_{camera}.mp4"
                try:
                    frames = render_segment_frames(self.env, clip, self.width, self.height,
                                                   camera=camera)
                    write_mp4(path, frames, self.fps)
                except Exception as exc:
                    errors.append(f"{label}/{camera}: {type(exc).__name__}: {exc}")
                    print(f"step={env_steps}: skipped pair video {path.name} ({exc})",
                          flush=True)
                    continue
                videos[label][camera] = str(path)
                paths.append(str(path))
        if not paths:
            raise RuntimeError("no pair videos written: "
                               + ("; ".join(errors) or "unknown error"))
        meta = {
            "index": index,
            "env_steps": env_steps,
            "feedback_steps": feedback_step,
            "cameras": list(self.cameras),
            "videos": videos,
            "errors": errors,
            "n_frames_a": int(len(clip_a.obs)),
            "n_frames_b": int(len(clip_b.obs)),
            **{key: record[key] for key in ("choice", "choice_probability", "probabilities",
                                            "gt_return_a_audit_only", "gt_return_b_audit_only",
                                            "skill_a", "skill_b", "keyframe_indices_a",
                                            "keyframe_indices_b") if key in record},
        }
        (self.directory / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"step={env_steps}: saved Jev pair videos ({', '.join(self.cameras)}): "
              f"{paths[0]} ...", flush=True)
        return paths
