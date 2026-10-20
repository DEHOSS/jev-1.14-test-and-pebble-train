"""Jev 1.14 preferences from a few camera frames, not trajectory text.

The request goes to the OpenRouter Decisions API, the same surface the
pebble-skill training used. Images are still PNG frames in time order.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from .envs import EnvSpec, is_metaworld, make_env, round_json_numbers
from .frames import check_frame_size, image_part, trajectory_frame_indices

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.14"
TIE_THRESHOLD = 0.60
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 520, 521, 522, 523, 524, 529}
PREFERENCE_QUESTION = {
    "type": "choice",
    "instructions": (
        "Compare the video `trajectory_A` with the video `trajectory_B` for the "
        "task in `task_description`. Each trajectory is a short sequence of "
        "camera frames in time order, earliest first. Decide only which "
        "trajectory better achieves or advances toward the task goal. Use the "
        "motion across those frames, not the names trajectory_A and "
        "trajectory_B. There is no numeric state and no environment reward. "
        "Treat `tie` as appropriate when the trajectories are effectively "
        "equal or the frames do not support a reliable preference."
    ),
    "criteria": {
        "A": ("Trajectory A more effectively achieves the goal or makes "
              "meaningfully better task-relevant progress."),
        "B": ("Trajectory B more effectively achieves the goal or makes "
              "meaningfully better task-relevant progress."),
        "tie": ("The trajectories are effectively equal, their difference is "
                "not task-relevant, or the frames are insufficient for a "
                "reliable preference."),
    },
}


class OpenRouterError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"{status}: {message}")


class OpenRouterConnectionError(Exception):
    pass


class Judgment:
    def __init__(self, choice: str, probabilities: dict[str, float], confidence: float,
                 target_b: float, text_a: str, text_b: str, raw_choice: str = ""):
        self.choice = choice
        self.raw_choice = raw_choice or choice
        self.probabilities = probabilities
        self.confidence = confidence
        self.target_b = target_b
        self.text_a = text_a
        self.text_b = text_b


def apply_tie_threshold(probabilities: dict[str, float], threshold: float) -> str:
    """Keep a tie only when P(tie) is strictly above the threshold."""
    if probabilities["tie"] > threshold:
        return "tie"
    if probabilities["A"] >= probabilities["B"]:
        return "A"
    return "B"


def _error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.strip() or "request failed"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or text)
    return text.strip() or "request failed"


def post_decisions(payload: dict, api_key: str, timeout: float) -> dict:
    data = json.dumps(round_json_numbers(payload)).encode("utf-8")
    request = urllib.request.Request(
        DECISIONS_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Jev image input is off unless this header is present.
            "TypeSafe-Danger-Zone-Images": "true",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        raise OpenRouterError(exc.code, _error_message(body) or exc.reason) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise OpenRouterConnectionError(str(exc)) from exc


class JevJudge:
    def __init__(self, spec: EnvSpec, mode: str = "probabilistic",
                 model: str = DEFAULT_MODEL, timeout: float = 120.0,
                 retry_delay_max: float = 60.0, api_key: str | None = None,
                 frames_per_clip: int = 8, camera: str = "corner",
                 width: int = 480, height: int = 480, seed: int = 0,
                 sweep_version: int = 3, tie_threshold: float = TIE_THRESHOLD):
        if mode not in ("probabilistic", "certain"):
            raise ValueError("mode must be 'probabilistic' or 'certain'")
        if not 0 <= tie_threshold <= 1:
            raise ValueError("tie threshold must be in [0, 1]")
        if timeout <= 0 or retry_delay_max <= 0:
            raise ValueError("timeout and retry delay must be positive")
        if frames_per_clip < 1:
            raise ValueError("frames_per_clip must be at least 1")
        check_frame_size(width, height)
        if is_metaworld(spec.name) and camera not in ("corner", "corner2"):
            raise ValueError("MetaWorld camera must be 'corner' or 'corner2'")
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY or pass --openrouter-api-key")
        self.spec, self.mode, self.model = spec, mode, model
        self.timeout, self.retry_delay_max, self.api_key = timeout, retry_delay_max, key
        self.frames_per_clip = frames_per_clip
        self.tie_threshold = tie_threshold
        self.camera = camera
        self.width = width
        self.height = height
        self.env = make_env(spec.name, seed, sweep_version)

    def _request_with_recovery(self, state: dict) -> dict:
        payload = {"model": self.model, "state": state,
                   "questions": {"preference": PREFERENCE_QUESTION}}
        attempt = 0
        while True:
            try:
                return post_decisions(payload, self.api_key, self.timeout)
            except OpenRouterConnectionError as exc:
                error = exc
            except OpenRouterError as exc:
                if exc.status not in TRANSIENT_STATUS and exc.status < 500:
                    raise
                error = exc
            attempt += 1
            delay = min(self.retry_delay_max, float(2 ** min(attempt, 6)))
            print(f"OpenRouter Decisions temporarily unavailable "
                  f"({type(error).__name__}: {error}); environment paused, "
                  f"retrying in {delay:g}s (attempt {attempt})",
                  file=sys.stderr, flush=True)
            time.sleep(delay)

    def _frames(self, segment) -> tuple[list[dict], list[int]]:
        if not segment.states:
            raise RuntimeError("segment has no simulator states to render for Jev 1.14")
        if len(segment.states) != len(segment.obs):
            raise RuntimeError("simulator states are not aligned with the segment")
        indices = trajectory_frame_indices(len(segment.states), self.frames_per_clip)
        parts = []
        for index in indices:
            self.env.set_sim_state(segment.states[index])
            frame = self.env.render_rgb(self.width, self.height, camera=self.camera)
            parts.append(image_part(frame))
        return parts, indices

    def _clip_state(self, parts: list[dict], indices: list[int], n_steps: int) -> dict:
        return {
            "time_order": "earliest frame first",
            "num_steps": int(n_steps),
            "frames": [
                {"step": int(index), "image": part}
                for index, part in zip(indices, parts)
            ],
        }

    def judge(self, a, b) -> Judgment:
        parts_a, idx_a = self._frames(a)
        parts_b, idx_b = self._frames(b)
        state = {
            "task_description": self.spec.task,
            "representation": "camera frames only",
            "trajectory_A": self._clip_state(parts_a, idx_a, len(a.obs)),
            "trajectory_B": self._clip_state(parts_b, idx_b, len(b.obs)),
        }
        response = self._request_with_recovery(state)
        answers = response.get("answers") or {}
        answer = answers.get("preference")
        if not isinstance(answer, dict):
            raise ValueError("OpenRouter returned no preference answer")
        probabilities = answer.get("probabilities") or {}
        try:
            probs = {key: float(probabilities[key]) for key in ("A", "B", "tie")}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("OpenRouter returned invalid probabilities") from exc
        if not np.all(np.isfinite(list(probs.values()))) or any(p < 0 for p in probs.values()):
            raise ValueError("OpenRouter returned invalid probabilities")
        total = sum(probs.values())
        if total <= 0:
            raise ValueError("OpenRouter returned zero probability mass")
        probs = {key: value / total for key, value in probs.items()}
        raw_choice = str(answer.get("choice", ""))
        if raw_choice not in probs:
            raise ValueError(f"OpenRouter returned unexpected choice {raw_choice!r}")
        choice = apply_tie_threshold(probs, self.tie_threshold)
        hard_target = {"A": 0.0, "B": 1.0, "tie": 0.5}[choice]
        if self.mode == "certain" or choice == "tie" or choice != raw_choice:
            target_b = hard_target
        else:
            target_b = probs["B"] + 0.5 * probs["tie"]
        confidence = float(answer.get("confidence", float("nan")))
        text_a = (f"jev-1.14 camera={self.camera} steps={len(a.obs)} "
                  f"frames={idx_a}")
        text_b = (f"jev-1.14 camera={self.camera} steps={len(b.obs)} "
                  f"frames={idx_b}")
        return Judgment(choice, probs, confidence, target_b, text_a, text_b,
                        raw_choice=raw_choice)

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def choice_probability(judgment: Judgment) -> float:
    return float(judgment.probabilities[judgment.choice])


def usable_preference(judgment: Judgment, min_choice_prob: float) -> bool:
    """Keep the label only if P(chosen A/B/tie) meets the floor. 0 disables."""
    if min_choice_prob <= 0:
        return True
    return choice_probability(judgment) >= min_choice_prob
