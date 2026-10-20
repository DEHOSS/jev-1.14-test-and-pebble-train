"""Save and restore the learned SAC, reward-model, and skill-detector weights."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch

from .learning import RewardEnsemble, SAC, SkillDetector, SkillReturnEstimator


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(run_dir: Path, agent: SAC, model: RewardEnsemble,
                    detector: SkillDetector, estimator: SkillReturnEstimator,
                    env_steps: int, feedback_steps: int, z_task: int = 0,
                    snapshots: Iterable[Path] | None = None) -> None:
    counters = {"env_steps": env_steps, "feedback_steps": feedback_steps, "z_task": z_task}
    sac = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "tq1": agent.tq1.state_dict(),
        "tq2": agent.tq2.state_dict(),
        "log_alpha": agent.log_alpha.detach().clone(),
        "n_skills": agent.n_skills,
        **counters,
    }
    reward = {
        "reward_members": [net.state_dict() for net in model.members],
        "reward_fitted": model.fitted,
        **counters,
    }
    skills = {
        "skill_method": detector.method,
        "skill_detector": detector.net.state_dict(),
        "skill_detector_fitted": detector.fitted,
        "skill_codebook": None if detector.codebook is None else detector.codebook.detach().cpu(),
        "skill_return": estimator.net.state_dict(),
        "skill_return_fitted": estimator.fitted,
        "n_skills": detector.n_skills,
        "sf_dim": detector.sf_dim,
        **counters,
    }
    full = {**sac, **reward, **skills}
    _atomic_save(sac, run_dir / "actor_critic.pt")
    _atomic_save(reward, run_dir / "reward_model.pt")
    _atomic_save(full, run_dir / "checkpoint.pt")
    for snapshot in snapshots or ():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(full, snapshot)


def load_model_weights(path: Path, agent: SAC, model: RewardEnsemble,
                       detector: SkillDetector | None = None,
                       estimator: SkillReturnEstimator | None = None) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.q1.load_state_dict(checkpoint["q1"])
    agent.q2.load_state_dict(checkpoint["q2"])
    agent.tq1.load_state_dict(checkpoint.get("tq1", checkpoint["q1"]))
    agent.tq2.load_state_dict(checkpoint.get("tq2", checkpoint["q2"]))
    if "log_alpha" in checkpoint:
        with torch.no_grad():
            agent.log_alpha.copy_(checkpoint["log_alpha"])
    members = checkpoint["reward_members"]
    if len(members) != len(model.members):
        raise ValueError(f"Checkpoint has {len(members)} reward members, model has {len(model.members)}")
    for network, state in zip(model.members, members):
        network.load_state_dict(state)
    model.fitted = bool(checkpoint.get("reward_fitted", True))
    if detector is not None and "skill_detector" in checkpoint:
        detector.net.load_state_dict(checkpoint["skill_detector"])
        detector.fitted = bool(checkpoint.get("skill_detector_fitted", True))
        codebook = checkpoint.get("skill_codebook")
        if codebook is not None and detector.codebook is not None:
            detector.codebook.copy_(torch.as_tensor(codebook, device=detector.device))
    if estimator is not None and "skill_return" in checkpoint:
        estimator.net.load_state_dict(checkpoint["skill_return"])
        estimator.fitted = bool(checkpoint.get("skill_return_fitted", True))
    return int(checkpoint["env_steps"]), int(checkpoint["feedback_steps"])
