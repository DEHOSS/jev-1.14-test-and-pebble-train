#!/usr/bin/env python3
"""Same 35 sweep-into pairs as the stride-40 test, with a tie threshold.

If the probability of tie is greater than 60%, the result is tie. Otherwise
the result is A or B, whichever probability is higher.

    export OPENROUTER_API_KEY=your_key
    python3 tresholded/run_sweep_preferences.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FRAMES_ROOT = Path(
    "/data/nazari/rlvlmf/training-cnn/jev-omni/eval_data/pebble_seed_12355/corner"
)
PAIR_JSON_DIR = Path(
    "/data/nazari/rlvlmf/jev-pebble-skill/runs_skill/sweep/probabilistic/"
    "feedback_1500/seed_12355/pair_videos"
)
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"
TASK = "Sweep the object into the target area with the robot end effector."
FRAMES_PER_SIDE = 8
TIE_THRESHOLD = 0.60
SOURCE_PAIRS = Path(__file__).resolve().parent.parent / "results_stride40.jsonl"
STORED_FRAMES = 10
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 520, 521, 522, 523, 524, 529}
QUESTION = {
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


def frame_slots(length: int, count: int) -> list[int]:
    """Same spacing as jev114.frames.trajectory_frame_indices."""
    count = min(length, count)
    if count == 1:
        return [0]
    return [int(round(i * (length - 1) / (count - 1))) for i in range(count)]


_PAIR_ID = re.compile(r"^step(?P<step>\d+)_fb(?P<fb>\d+)_pair(?P<pair>\d+)$")


def pair_order(pair_id: str) -> tuple[int, int, int, str]:
    match = _PAIR_ID.match(pair_id)
    if match is None:
        return (10**12, 10**12, 10**12, pair_id)
    return (int(match.group("step")), int(match.group("fb")), int(match.group("pair")), pair_id)


def discover_pairs(frames_root: Path) -> list[str]:
    sides: dict[str, set[str]] = {}
    for path in frames_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name.endswith("_A_corner"):
            sides.setdefault(name[: -len("_A_corner")], set()).add("A")
        elif name.endswith("_B_corner"):
            sides.setdefault(name[: -len("_B_corner")], set()).add("B")
    complete = [pair_id for pair_id, found in sides.items() if found == {"A", "B"}]
    return sorted(complete, key=pair_order)


def select_feedback_stride(pair_ids: list[str], stride: int) -> list[str]:
    """One pair from each pause nearest to feedback 1, 1+stride, 1+2*stride, ...

    Saved pauses begin at feedback 1, 16, 31, ... and each stores pair 01
    through 05. The pair kept at a pause is pair 01.
    """
    if stride < 1:
        raise ValueError("feedback stride must be at least 1")
    by_pause: dict[int, list[str]] = {}
    last_feedback = 0
    for pair_id in pair_ids:
        match = _PAIR_ID.match(pair_id)
        if match is None:
            continue
        feedback = int(match.group("fb"))
        last_feedback = max(last_feedback, feedback)
        pause = 1 + ((feedback - 1) // 15) * 15
        by_pause.setdefault(pause, []).append(pair_id)
    if not by_pause or last_feedback < 1:
        return []
    pauses = sorted(by_pause)
    chosen = []
    for target in range(1, last_feedback + 1, stride):
        pause = min(pauses, key=lambda start: (abs(start - target), start))
        pairs = sorted(by_pause[pause], key=pair_order)
        first = next((pair_id for pair_id in pairs if pair_id.endswith("_pair01")), pairs[0])
        if first not in chosen:
            chosen.append(first)
    return chosen


def image_part(path: Path) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def clip_state(folder: Path) -> dict:
    meta = json.loads((folder / "frames.json").read_text(encoding="utf-8"))
    source_steps = [int(step) for step in meta["frame_indices"]]
    if len(source_steps) != STORED_FRAMES:
        raise ValueError(f"{folder.name} has {len(source_steps)} source steps, expected {STORED_FRAMES}")
    slots = frame_slots(STORED_FRAMES, FRAMES_PER_SIDE)
    frames = []
    for slot in slots:
        picture = folder / f"frame_{slot:03d}.png"
        if not picture.is_file():
            raise FileNotFoundError(picture)
        frames.append({"step": source_steps[slot], "image": image_part(picture)})
    return {
        "time_order": "earliest frame first",
        "num_steps": int(source_steps[-1]) + 1 if source_steps else 0,
        "frames": frames,
        "shown_source_steps": [frame["step"] for frame in frames],
    }


def post_decisions(payload: dict, api_key: str, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        DECISIONS_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code}: {body.strip() or exc.reason}") from exc


def judge_pair(state: dict, api_key: str, timeout: float, retry_delay_max: float) -> dict:
    payload = {"model": MODEL, "state": state, "questions": {"preference": QUESTION}}
    attempt = 0
    while True:
        try:
            response = post_decisions(payload, api_key, timeout)
            break
        except (RuntimeError, urllib.error.URLError, TimeoutError, OSError) as exc:
            status = None
            text = str(exc)
            if text.split(":", 1)[0].isdigit():
                status = int(text.split(":", 1)[0])
            if status is not None and status not in TRANSIENT_STATUS and status < 500:
                raise
            attempt += 1
            delay = min(retry_delay_max, float(2 ** min(attempt, 6)))
            print(f"  retry in {delay:g}s after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            time.sleep(delay)
    answer = (response.get("answers") or {}).get("preference")
    if not isinstance(answer, dict):
        raise RuntimeError(f"no preference answer: {response}")
    probabilities = answer.get("probabilities") or {}
    probs = {key: float(probabilities[key]) for key in ("A", "B", "tie")}
    total = sum(probs.values())
    if total <= 0:
        raise RuntimeError(f"zero probability mass: {answer}")
    probs = {key: value / total for key, value in probs.items()}
    choice = str(answer.get("choice", ""))
    if choice not in probs:
        raise RuntimeError(f"unexpected choice {choice!r}: {answer}")
    usage = response.get("usage") or {}
    return {
        "raw_choice": choice,
        "choice": apply_tie_threshold(probs, TIE_THRESHOLD),
        "probabilities": probs,
        "confidence": answer.get("confidence"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost": usage.get("cost"),
    }


def apply_tie_threshold(probabilities: dict[str, float], threshold: float) -> str:
    """Return tie only when its probability is above the threshold."""
    if probabilities["tie"] > threshold:
        return "tie"
    if probabilities["A"] >= probabilities["B"]:
        return "A"
    return "B"


def exact_stride40_pairs() -> list[str]:
    if not SOURCE_PAIRS.is_file():
        raise FileNotFoundError(f"Missing the stride-40 pair list: {SOURCE_PAIRS}")
    pair_ids = []
    for line in SOURCE_PAIRS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pair_ids.append(json.loads(line)["pair_id"])
    if len(pair_ids) != 35:
        raise RuntimeError(f"Expected 35 pairs in {SOURCE_PAIRS}, found {len(pair_ids)}")
    return pair_ids


def ground_truth(pair_id: str) -> tuple[str, float, float]:
    meta = json.loads((PAIR_JSON_DIR / f"{pair_id}.json").read_text(encoding="utf-8"))
    return_a = float(meta["gt_return_a_audit_only"])
    return_b = float(meta["gt_return_b_audit_only"])
    if return_a > return_b:
        winner = "A"
    elif return_b > return_a:
        winner = "B"
    else:
        winner = "tie"
    return winner, return_a, return_b


def completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["pair_id"])
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=10, help="How many earliest A/B pairs to judge")
    parser.add_argument("--feedback-stride", type=int, default=0,
                        help="If set, judge pair 01 at checkpoints this many feedback labels apart")
    parser.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retry-delay-max", type=float, default=60.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY first.", file=sys.stderr)
        return 1
    if args.pairs < 1:
        print("--pairs must be at least 1", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    if args.output is None:
        args.output = here / "results.jsonl"
    try:
        selected = exact_stride40_pairs()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    done = completed(args.output)
    todo = [pair_id for pair_id in selected if pair_id not in done]
    print(f"task: {TASK}")
    print(f"model: {MODEL}")
    print(f"tie threshold: P(tie) > {TIE_THRESHOLD:.0%} otherwise A or B")
    print(f"pairs: {len(selected)}  already done: {len(selected) - len(todo)}  to run: {len(todo)}")
    for pair_id in selected:
        mark = "done" if pair_id in done else "todo"
        print(f"  [{mark}] {pair_id}")
    if not todo:
        print(f"nothing to run; results are in {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    judged = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for index, pair_id in enumerate(todo, start=1):
            folder_a = args.frames_root / f"{pair_id}_A_corner"
            folder_b = args.frames_root / f"{pair_id}_B_corner"
            clip_a = clip_state(folder_a)
            clip_b = clip_state(folder_b)
            state = {
                "task_description": TASK,
                "representation": "camera frames only",
                "trajectory_A": {key: value for key, value in clip_a.items() if key != "shown_source_steps"},
                "trajectory_B": {key: value for key, value in clip_b.items() if key != "shown_source_steps"},
            }
            print(f"[{index}/{len(todo)}] {pair_id}", flush=True)
            judgment = judge_pair(state, api_key, args.timeout, args.retry_delay_max)
            truth, return_a, return_b = ground_truth(pair_id)
            hit = judgment["choice"] == truth
            judged += 1
            correct += int(hit)
            record = {
                "pair_id": pair_id,
                "task_description": TASK,
                "frames_per_side": FRAMES_PER_SIDE,
                "source_steps_a": clip_a["shown_source_steps"],
                "source_steps_b": clip_b["shown_source_steps"],
                "raw_choice": judgment["raw_choice"],
                "tie_threshold": TIE_THRESHOLD,
                "choice": judgment["choice"],
                "probabilities": judgment["probabilities"],
                "confidence": judgment["confidence"],
                "ground_truth": truth,
                "correct": hit,
                "gt_return_a": return_a,
                "gt_return_b": return_b,
                "input_tokens": judgment["input_tokens"],
                "output_tokens": judgment["output_tokens"],
                "cost": judgment["cost"],
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"  raw={judgment['raw_choice']}  jev={judgment['choice']}  "
                f"P(tie)={judgment['probabilities']['tie']:.2f}  truth={truth}  "
                f"match={hit}  returns A={return_a:.3f} B={return_b:.3f}  "
                f"tokens={judgment['input_tokens']}",
                flush=True,
            )
    print(f"matched ground truth on {correct}/{judged} new pairs")
    print(f"results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
