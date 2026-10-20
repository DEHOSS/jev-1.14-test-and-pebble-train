"""PEBBLE + S-EPOA skill-conditioned SAC with Jev 1.14 video preferences.

Same training loop as jev-pebble-skill. The judge is typesafe/jev-1.14 on the
OpenRouter Decisions API. Each query sends a few camera frames in time order
instead of a textual observation/action trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import deque
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .envs import ENV_CHOICES, is_metaworld, make_env
from .frames import check_frame_size
from .judge import JevJudge, choice_probability, usable_preference
from .keyframes import subsample_segment
from .learning import (EpisodeStore, Preference, ReplayBuffer, RewardEnsemble, SAC,
                       SkillDetector, SkillReturnEstimator, select_preference_pair)
from .video import PairVideoSaver


def scaled_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + (action + 1.0) * 0.5 * (high - low)


class CSVLog:
    def __init__(self, path: Path, fields: list[str]):
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def write(self, **values):
        self.writer.writerow(values)

    def close(self):
        self.file.close()


def evaluate(env, agent: SAC, model: RewardEnsemble, episodes: int,
             skill) -> dict[str, float]:
    gt_returns, learned_returns, successes = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        observations, actions = [], []
        gt_return, success = 0.0, False
        for _ in range(env.spec.max_episode_steps):
            action = agent.act(obs, skill, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(
                scaled_action(action, env.spec.action_low, env.spec.action_high))
            observations.append(obs)
            actions.append(action)
            gt_return += reward
            success = success or bool(info.get("success", False))
            obs = next_obs
            if terminated or truncated:
                break
        gt_returns.append(gt_return)
        learned_returns.append(float(model.predict(np.stack(observations), np.stack(actions)).sum())
                               if model.fitted else float("nan"))
        successes.append(float(success))
    return {"gt_return_mean": float(np.mean(gt_returns)),
            "learned_return_mean": float(np.mean(learned_returns)),
            "success_rate": float(np.mean(successes)) if is_metaworld(env.spec.name) else float("nan")}


def request_feedback(store: EpisodeStore, model: RewardEnsemble, judge: JevJudge,
                     count: int, rng: np.random.Generator, dataset: list[Preference],
                     log_file, env_steps: int, initial_count: int, candidates: int,
                     query_attempts: int, disagreement_percentile: float,
                     disagreement_history: deque[float], detector: SkillDetector,
                     estimator: SkillReturnEstimator, min_skill_tv: float,
                     min_choice_prob: float, keyframe_length: int = 0,
                     pair_videos: PairVideoSaver | None = None) -> int:
    accepted = 0
    skipped = 0
    consecutive_skips = 0
    if pair_videos is not None:
        pair_videos.begin_query()
    while accepted < count:
        a, b, query = select_preference_pair(
            store, model, rng, candidates, query_attempts, disagreement_percentile,
            disagreement_history, detector, estimator, min_skill_tv)
        idx_a = idx_b = None
        if keyframe_length and (len(a.obs) > keyframe_length or len(b.obs) > keyframe_length):
            a, idx_a = subsample_segment(a, keyframe_length)
            b, idx_b = subsample_segment(b, keyframe_length)
        pair = (a, b)
        judgment = judge.judge(*pair)
        used = usable_preference(judgment, min_choice_prob)
        if used:
            dataset.append(Preference(pair[0], pair[1], judgment.target_b))
            accepted += 1
            consecutive_skips = 0
            feedback_step = initial_count + accepted
        else:
            skipped += 1
            consecutive_skips += 1
            feedback_step = initial_count + accepted
            if consecutive_skips >= 32:
                raise RuntimeError(
                    f"Jev P(choice) stayed below {min_choice_prob:g} for 32 queries; "
                    "lower --jev-min-choice-prob or inspect the pairs")
        record = {
            "env_steps": env_steps, "feedback_steps": feedback_step,
            "used": used,
            "choice": judgment.choice, "raw_choice": judgment.raw_choice,
            "tie_threshold": judge.tie_threshold,
            "choice_probability": choice_probability(judgment),
            "min_choice_prob": min_choice_prob,
            "probabilities": judgment.probabilities,
            "confidence": judgment.confidence, "target_b": judgment.target_b,
            "disagreement": query["disagreement"],
            "disagreement_threshold": query["disagreement_threshold"],
            "skill_tv": query["skill_tv"], "skill_gap": query["skill_gap"],
            "sepoa_score": query["sepoa_score"],
            "skill_a": query["skill_a"], "skill_b": query["skill_b"],
            "keyframe_indices_a": None if idx_a is None else idx_a.tolist(),
            "keyframe_indices_b": None if idx_b is None else idx_b.tolist(),
            "gt_return_a_audit_only": pair[0].gt_return,
            "gt_return_b_audit_only": pair[1].gt_return,
            "gt_gap_audit_only": abs(pair[0].gt_return - pair[1].gt_return),
            "trajectory_a": judgment.text_a, "trajectory_b": judgment.text_b,
        }
        if used and pair_videos is not None:
            record["pair_videos"] = pair_videos.maybe_save(
                pair[0], pair[1], env_steps, feedback_step, record)
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if used and (feedback_step % 10 == 0 or accepted == count):
            print(f"step={env_steps}: Jev feedback {feedback_step} received", flush=True)
        elif not used:
            print(f"step={env_steps}: skipped Jev label "
                  f"P({judgment.choice})={choice_probability(judgment):.3f} "
                  f"< {min_choice_prob:g} (skipped={skipped})", flush=True)
    if skipped:
        print(f"step={env_steps}: skipped {skipped} low-probability Jev labels "
              f"in this batch", flush=True)
    return initial_count + accepted


def eval_rank(metrics: dict) -> tuple[float, float]:
    success = metrics["success_rate"]
    gt = float(metrics["gt_return_mean"])
    if not math.isfinite(success):
        success = 0.0
    if not math.isfinite(gt):
        gt = float("-inf")
    return (success, gt)


def train(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    positive = ("feedback_budget", "feedback_batch", "query_interval", "warmup_steps",
                "pretrain_steps", "steps", "candidates", "ensemble", "hidden",
                "reward_updates", "reward_batch", "replay_size", "batch_size",
                "updates_per_step", "eval_interval", "eval_episodes",
                "checkpoint_interval", "jev_timeout", "jev_retry_delay_max")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.segment_length <= 0 or args.segment_length > 500:
        raise ValueError("segment length must be between 1 and 500")
    if args.keyframe_length < 0:
        raise ValueError("keyframe_length cannot be negative")
    if args.keyframe_length > args.segment_length:
        raise ValueError("keyframe_length cannot exceed segment_length")
    if args.warmup_steps > args.pretrain_steps:
        raise ValueError("warmup_steps cannot exceed pretrain_steps")
    if args.query_attempts <= 0 or args.candidates <= 0:
        raise ValueError("query attempts and candidates must be positive")
    if args.disagreement_percentile < 0 or args.disagreement_percentile > 100:
        raise ValueError("disagreement percentile must be in [0, 100]")
    if args.disagreement_window <= 0:
        raise ValueError("disagreement window must be positive")
    if args.query_min_learned_gap < 0:
        raise ValueError("query thresholds cannot be negative")
    if args.jev_min_choice_prob < 0 or args.jev_min_choice_prob > 1:
        raise ValueError("Jev min choice probability must be in [0, 1]")
    if args.jev_tie_threshold < 0 or args.jev_tie_threshold > 1:
        raise ValueError("Jev tie threshold must be in [0, 1]")
    if args.n_skills < 2:
        raise ValueError("n_skills must be at least 2")
    if args.sf_dim < 2:
        raise ValueError("sf_dim must be at least 2")
    if args.aps_horizon <= 0:
        raise ValueError("aps_horizon must be positive")
    if args.min_skill_tv < 0 or args.min_skill_tv > 1:
        raise ValueError("min skill TV must be in [0, 1]")
    if args.pair_videos < 0:
        raise ValueError("pair_videos cannot be negative")
    if args.frames_per_clip < 1:
        raise ValueError("frames_per_clip must be at least 1")
    check_frame_size(args.render_width, args.render_height)
    rounds = math.ceil(args.feedback_budget / args.feedback_batch)
    last_query_step = args.pretrain_steps + (rounds - 1) * args.query_interval
    if args.steps < last_query_step:
        raise ValueError(f"{args.steps} steps cannot exhaust the feedback budget; need at least "
                         f"{last_query_step} with the current schedule")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env, args.seed, args.sweep_version)
    eval_env = make_env(args.env, args.seed + 10000, args.sweep_version)
    render_env = None
    if args.pair_videos:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; continuing without pair videos", flush=True)
        else:
            try:
                render_env = make_env(args.env, args.seed + 20000, args.sweep_version)
            except Exception as exc:
                print(f"Could not create pair-video env ({type(exc).__name__}: {exc}); "
                      "continuing without pair videos", flush=True)
    judge = JevJudge(env.spec, args.mode, args.jev_model,
                     args.jev_timeout, args.jev_retry_delay_max,
                     args.openrouter_api_key,
                     frames_per_clip=args.frames_per_clip,
                     camera=args.camera,
                     width=args.render_width, height=args.render_height,
                     seed=args.seed + 30000, sweep_version=args.sweep_version,
                     tie_threshold=args.jev_tie_threshold)
    run_dir = Path(args.output) / args.env / args.mode / f"feedback_{args.feedback_budget}" / f"seed_{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    if config.get("openrouter_api_key"):
        config["openrouter_api_key"] = "<redacted>"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    obs_dim, act_dim = len(env.spec.obs_names), len(env.spec.action_names)
    replay = ReplayBuffer(args.replay_size, obs_dim, act_dim)
    store = EpisodeStore(args.segment_length)
    model = RewardEnsemble(obs_dim, act_dim, device, args.ensemble, args.hidden)
    detector = SkillDetector(obs_dim, args.n_skills, device, args.hidden,
                             method=args.skill_method, sf_dim=args.sf_dim,
                             knn_k=args.knn_k)
    estimator = SkillReturnEstimator(args.n_skills, device)
    agent = SAC(obs_dim, act_dim, args.n_skills, device, args.hidden,
                skill_dim=detector.skill_dim,
                successor_critic=args.skill_method == "aps")
    dataset: list[Preference] = []
    reward_log = CSVLog(run_dir / "rewards.csv", ["env_steps", "feedback_steps", "gt_reward",
                                                    "learned_reward", "episode"])
    eval_log = CSVLog(run_dir / "evaluation.csv", ["env_steps", "feedback_steps",
                                                     "gt_return_mean", "learned_return_mean",
                                                     "success_rate", "reward_loss",
                                                     "reward_accuracy"])
    feedback_log = (run_dir / "feedback.jsonl").open("w", encoding="utf-8")
    disagreement_history: deque[float] = deque(maxlen=args.disagreement_window)
    obs, _ = env.reset()
    episode, episode_length, feedback_steps = 0, 0, 0
    skill_id = int(rng.integers(args.n_skills))
    skill_vec = detector.skill_vector(skill_id)
    z_task = 0
    reward_loss, reward_accuracy = float("nan"), float("nan")
    best_rank = (float("-inf"), float("-inf"))
    ckpt_dir = run_dir / "checkpoints"
    pair_videos = None
    if render_env is not None:
        pair_videos = PairVideoSaver(render_env, run_dir / "pair_videos",
                                     args.pair_videos)
    start = time.monotonic()
    try:
        for step in range(1, args.steps + 1):
            if step <= args.warmup_steps:
                action = rng.uniform(-1, 1, size=act_dim).astype(np.float32)
            else:
                action = agent.act(obs, skill_vec)
            try:
                sim_state = env.get_sim_state()
            except Exception as exc:
                raise RuntimeError(
                    f"step={step}: could not read the simulator state Jev 1.14 "
                    f"needs in order to render frames ({type(exc).__name__}: {exc})"
                ) from exc
            next_obs, gt_reward, terminated, truncated, info = env.step(
                scaled_action(action, env.spec.action_low, env.spec.action_high))
            episode_length += 1
            done = terminated or truncated or episode_length >= env.spec.max_episode_steps
            terminal = float(terminated and not truncated and episode_length < env.spec.max_episode_steps)
            learned = (float(model.predict(obs[None], action[None])[0])
                       if model.fitted else float("nan"))
            replay.add(obs, action, gt_reward, next_obs, terminal, learned, skill_id)
            store.add(obs, action, gt_reward, done, skill_id, state=sim_state)
            reward_log.write(env_steps=step, feedback_steps=feedback_steps,
                             gt_reward=gt_reward, learned_reward=learned, episode=episode)
            obs = next_obs
            resample = False
            if done:
                obs, _ = env.reset()
                episode += 1
                episode_length = 0
                resample = True
            elif (args.skill_method == "aps" and step < args.pretrain_steps
                  and step % args.aps_horizon == 0):
                resample = True
            if resample:
                if step < args.pretrain_steps:
                    skill_id = int(rng.integers(args.n_skills))
                else:
                    skill_id = z_task
                skill_vec = detector.skill_vector(skill_id)

            query_due = (step >= args.pretrain_steps and
                         (step - args.pretrain_steps) % args.query_interval == 0 and
                         feedback_steps < args.feedback_budget)
            if query_due:
                n = min(args.feedback_batch, args.feedback_budget - feedback_steps)
                print(f"step={step}: waiting for {n} Jev preferences; environment paused",
                      flush=True)
                if model.fitted and replay.size:
                    idx = rng.integers(replay.size, size=min(256, replay.size))
                    estimator.update(replay.skill[idx], replay.learned[idx])
                    z_task = estimator.best_skill()
                    skill_id = z_task
                    skill_vec = detector.skill_vector(skill_id)
                feedback_steps = request_feedback(store, model, judge, n, rng, dataset,
                                                  feedback_log, step, feedback_steps,
                                                  args.candidates, args.query_attempts,
                                                  args.disagreement_percentile,
                                                  disagreement_history, detector, estimator,
                                                  args.min_skill_tv, args.jev_min_choice_prob,
                                                  args.keyframe_length, pair_videos)
                reward_loss, reward_accuracy = model.fit(
                    dataset, rng, args.reward_updates, args.reward_batch)
                replay.relabel(model)
                if replay.size:
                    idx = rng.integers(replay.size, size=min(256, replay.size))
                    estimator.update(replay.skill[idx], replay.learned[idx])
                    z_task = estimator.best_skill()
                    skill_id = z_task
                    skill_vec = detector.skill_vector(skill_id)
                print(f"step={step}: feedback complete ({feedback_steps}/{args.feedback_budget}); "
                      f"reward_loss={reward_loss:.4f} reward_acc={reward_accuracy:.3f} "
                      f"z_task={z_task}; resuming environment", flush=True)
            if replay.size >= args.batch_size:
                obs_b, act_b, learned_b, next_obs_b, terminal_b, skill_b = replay.sample(
                    args.batch_size, rng, device)
                skill_z = detector.skill_tensor(skill_b)
                if args.skill_method == "aps":
                    detector.update(obs_b, skill_b, next_obs=next_obs_b)
                else:
                    detector.update(obs_b, skill_b)
                if step < args.pretrain_steps:
                    if args.skill_method == "aps":
                        memory = torch.as_tensor(replay.neighbor_obs(rng, 512), device=device)
                        reward_b = detector.aps_rewards(next_obs_b, skill_b, memory)
                    else:
                        reward_b = detector.diayn_rewards(obs_b, skill_b)
                    for _ in range(args.updates_per_step):
                        agent.update((obs_b, act_b, reward_b, next_obs_b, terminal_b, skill_z))
                elif model.fitted:
                    for _ in range(args.updates_per_step):
                        agent.update((obs_b, act_b, learned_b, next_obs_b, terminal_b, skill_z))
            snapshots: list[Path] = []
            evaluated = False
            if step % args.eval_interval == 0 or step == args.steps:
                metrics = evaluate(eval_env, agent, model, args.eval_episodes,
                                   detector.skill_vector(z_task))
                evaluated = True
                eval_log.write(env_steps=step, feedback_steps=feedback_steps,
                               reward_loss=reward_loss, reward_accuracy=reward_accuracy,
                               **metrics)
                reward_log.file.flush()
                eval_log.file.flush()
                print(f"{args.env} {args.mode} budget={args.feedback_budget} seed={args.seed} "
                      f"step={step} feedback={feedback_steps} gt={metrics['gt_return_mean']:.2f} "
                      f"learned={metrics['learned_return_mean']:.2f} "
                      f"success={metrics['success_rate']:.3f} "
                      f"reward_loss={reward_loss:.4f} reward_acc={reward_accuracy:.3f} "
                      f"elapsed={time.monotonic() - start:.0f}s", flush=True)
            if step % args.checkpoint_interval == 0 or step == args.steps:
                snapshots.append(ckpt_dir / f"step_{step:07d}.pt")
            if evaluated:
                rank = eval_rank(metrics)
                if rank > best_rank:
                    best_rank = rank
                    snapshots.append(ckpt_dir / f"best_step_{step:07d}.pt")
                    snapshots.append(ckpt_dir / "best.pt")
                    print(f"step={step}: new best checkpoint "
                          f"success={metrics['success_rate']:.3f} "
                          f"gt={metrics['gt_return_mean']:.2f} -> {snapshots[-2]}",
                          flush=True)
            if query_due or snapshots:
                save_checkpoint(run_dir, agent, model, detector, estimator, step,
                                feedback_steps, z_task, snapshots=snapshots)
        if feedback_steps != args.feedback_budget:
            raise RuntimeError("Training ended before consuming the requested feedback budget")
    finally:
        reward_log.close()
        eval_log.close()
        feedback_log.close()
        judge.close()
        env.close()
        eval_env.close()
        if render_env is not None:
            render_env.close()

    from .plot import plot_run
    plots = plot_run(run_dir, args.env)
    print(f"Saved {len(plots)} plots to {run_dir / 'plots'}", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=ENV_CHOICES, default="sweep")
    p.add_argument("--mode", choices=("certain", "probabilistic"), default="probabilistic")
    p.add_argument("--feedback-budget", type=int, default=1500,
                   help="Number of successful Jev preferences; small values work for smoke tests")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", default="runs_jev114")
    p.add_argument("--device", default="auto")
    p.add_argument("--jev-model", default="typesafe/jev-1.14",
                   help="OpenRouter Decisions model id")
    p.add_argument("--openrouter-api-key", default=None,
                   help="OpenRouter API key; otherwise OPENROUTER_API_KEY is used")
    p.add_argument("--frames-per-clip", type=int, default=8,
                   help="Camera frames sent for each clip, earliest to latest, "
                        "spaced across the segment. Numeric state is not sent")
    p.add_argument("--camera", default="corner",
                   help="MetaWorld camera rendered for Jev. Ignored by dm_control")
    p.add_argument("--render-width", type=int, default=480)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--jev-timeout", type=float, default=120.0,
                   help="Timeout in seconds for each OpenRouter Decisions request")
    p.add_argument("--jev-retry-delay-max", type=float, default=60.0,
                   help="Maximum seconds between retries of transient OpenRouter errors")
    p.add_argument("--sweep-version", type=int, choices=(2, 3), default=3)
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Random-action steps at the start of unsupervised pretraining")
    p.add_argument("--pretrain-steps", type=int, default=10_000,
                   help="Unsupervised skill-discovery steps before Jev")
    p.add_argument("--n-skills", type=int, default=8,
                   help="Number of discrete skills (APS codebook size or DIAYN one-hots)")
    p.add_argument("--skill-method", choices=("aps", "diayn"), default="aps",
                   help="Unsupervised skill discovery: APS (S-EPOA default) or DIAYN")
    p.add_argument("--sf-dim", type=int, default=16,
                   help="APS successor-feature dimension")
    p.add_argument("--aps-horizon", type=int, default=50,
                   help="Resample the APS skill vector this often during pretrain (steps)")
    p.add_argument("--min-skill-tv", type=float, default=0.25,
                   help="Reject pairs whose skill-posterior total variation is below this")
    p.add_argument("--knn-k", type=int, default=12,
                   help="k for APS/APT particle entropy (ignored by DIAYN)")
    p.add_argument("--query-interval", type=int, default=4000)
    p.add_argument("--feedback-batch", type=int, default=15,
                   help="Jev labels per pause. With budget 1500, interval 4000, and "
                        "10k pretrain, 15 labels finishes the last query at step 406k")
    p.add_argument("--segment-length", type=int, default=50)
    p.add_argument("--keyframe-length", type=int, default=0,
                   help="0 renders frames from the full segment, then keeps "
                        "--frames-per-clip of them. A positive value first compresses "
                        "each clip to that many salient simulator states")
    p.add_argument("--candidates", type=int, default=32,
                   help="Unlabeled pairs scored by ensemble disagreement per draw")
    p.add_argument("--query-attempts", type=int, default=32,
                   help="Redraw candidate sets this many times when none meet the percentile")
    p.add_argument("--disagreement-percentile", type=float, default=90.0,
                   help="Keep pairs at or above this percentile of recent candidate "
                        "disagreement (0 disables the floor and uses classic argmax-style "
                        "sampling among the current draw)")
    p.add_argument("--disagreement-window", type=int, default=4096,
                   help="Number of recent candidate disagreement scores used for the percentile")
    p.add_argument("--query-min-learned-gap", type=float, default=0.0,
                   help="Unused in the skill variant; kept for config compatibility")
    p.add_argument("--jev-min-choice-prob", type=float, default=0.0,
                   help="Ignore a Jev label when P(chosen A/B/tie) is below this (0 disables). "
                        "Redraws another pair until the batch is filled with usable labels")
    p.add_argument("--jev-tie-threshold", type=float, default=0.60,
                   help="Keep a tie only when P(tie) is greater than this. "
                        "Otherwise the label is the stronger of A or B")
    p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reward-updates", type=int, default=100)
    p.add_argument("--reward-batch", type=int, default=32)
    p.add_argument("--replay-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    p.add_argument("--pair-videos", type=int, default=1,
                   help="Accepted Jev pairs to save as clip-A/clip-B MP4s at each query "
                        "pause (0 disables). Default 1 is the first label of every pause, "
                        "from corner2 and corner")
    return p


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
