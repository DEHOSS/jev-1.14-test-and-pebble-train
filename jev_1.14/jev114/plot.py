"""Plot training rewards or evaluation returns against environment or feedback steps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .envs import ENV_CHOICES, is_metaworld


TRAIN_METRICS = ("gt_reward", "learned_reward")
EVAL_METRICS = ("gt_return_mean", "learned_return_mean", "success_rate",
                "reward_loss", "reward_accuracy")
OVERLAY_GROUPS = (
    ("rewards.csv", ("gt_reward", "learned_reward"), "reward",
     "ground truth", "learned"),
    ("evaluation.csv", ("gt_return_mean", "learned_return_mean"), "return",
     "ground truth", "learned"),
)
SINGLE_EVAL_METRICS = ("reward_loss", "reward_accuracy", "success_rate")


def feedback_plot_cutoff(config: dict, metric: str) -> int | None:
    """Keep only the first comparable window after feedback is exhausted."""
    query_start = config.get("pretrain_steps", config.get("warmup_steps"))
    required = ("query_interval", "feedback_budget", "feedback_batch", "eval_interval")
    if query_start is None or any(name not in config for name in required):
        return None  # Older runs without a complete schedule retain their original plots.
    rounds = math.ceil(config["feedback_budget"] / config["feedback_batch"])
    final_query_step = query_start + (rounds - 1) * config["query_interval"]
    if metric in TRAIN_METRICS:
        return final_query_step + config["query_interval"]
    return final_query_step + config["eval_interval"] - 1


def load_series(path: Path, metric: str, axis: str, bin_size: int,
                max_env_steps: int | None = None) -> dict[int, float]:
    """Mean values per x-axis bin, optionally ending shortly after final feedback."""
    buckets: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if max_env_steps is not None and int(row["env_steps"]) > max_env_steps:
                continue
            value = float(row[metric])
            if not math.isfinite(value):
                continue  # No learned reward exists before the first feedback round.
            position = int(row[axis])
            if metric in TRAIN_METRICS:
                position = position // bin_size * bin_size
            buckets.setdefault(position, []).append(value)
    return {position: sum(values) / len(values) for position, values in buckets.items()}


def _bin_size(config: dict, axis: str) -> int:
    default_bin = 10000 if axis == "env_steps" else 100
    total = config.get("steps" if axis == "env_steps" else "feedback_budget")
    return min(default_bin, max(1, math.ceil(total / 100))) if total else default_bin


def _cutoff(config: dict, metric: str, axis: str) -> int | None:
    return feedback_plot_cutoff(config, metric) if axis == "feedback_steps" else None


def _draw_overlay(ax, gt_series: dict[int, float], learned_series: dict[int, float],
                  gt_label: str, learned_label: str):
    ax2 = ax.twinx()
    lines = []
    if gt_series:
        x = sorted(gt_series)
        lines += ax.plot(x, [gt_series[step] for step in x], marker="o", markersize=3,
                         color="C0", label=gt_label)
    if learned_series:
        x = sorted(learned_series)
        lines += ax2.plot(x, [learned_series[step] for step in x], marker="o", markersize=3,
                          color="C1", label=learned_label)
    if lines:
        ax.legend(lines, [line.get_label() for line in lines], loc="best")
    return ax2


def plot_run(run_dir: Path, env: str) -> list[Path]:
    """Save every reward/return curve for one completed training run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    created = []

    for source_name, (gt_metric, learned_metric), stem, gt_label, learned_label in OVERLAY_GROUPS:
        source = run_dir / source_name
        for axis in ("env_steps", "feedback_steps"):
            bin_size = _bin_size(config, axis)
            gt_series = load_series(source, gt_metric, axis, bin_size,
                                    _cutoff(config, gt_metric, axis))
            learned_series = load_series(source, learned_metric, axis, bin_size,
                                         _cutoff(config, learned_metric, axis))
            if not gt_series and not learned_series:
                raise ValueError(f"No finite {gt_metric}/{learned_metric} values in {source}")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax2 = _draw_overlay(ax, gt_series, learned_series, gt_label, learned_label)
            ax.set_xlabel(axis.replace("_", " "))
            ax.set_ylabel(gt_label)
            ax2.set_ylabel(learned_label)
            ax.set_title(f"{env}: ground truth and learned {stem}")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            path = output_dir / f"{stem}_vs_{axis}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(path)

    singles = ("reward_loss", "reward_accuracy") + (("success_rate",) if is_metaworld(env) else ())
    for metric in singles:
        source = run_dir / "evaluation.csv"
        for axis in ("env_steps", "feedback_steps"):
            series = load_series(source, metric, axis, _bin_size(config, axis),
                                 _cutoff(config, metric, axis))
            if not series:
                raise ValueError(f"No finite {metric} values in {source}")
            x = sorted(series)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, [series[step] for step in x], marker="o", markersize=3)
            ax.set(xlabel=axis.replace("_", " "), ylabel=metric.replace("_", " "),
                   title=f"{env}: {metric.replace('_', ' ')}")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            path = output_dir / f"{metric}_vs_{axis}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(path)
    return created


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--env", choices=ENV_CHOICES, required=True)
    p.add_argument("--metric", choices=("reward", "return") + SINGLE_EVAL_METRICS +
                   TRAIN_METRICS + EVAL_METRICS[:2], default="reward")
    p.add_argument("--axis", choices=("env_steps", "feedback_steps"), default="env_steps")
    p.add_argument("--bin-size", type=int, default=None,
                   help="Training reward bin width; default 10000 env or 100 feedback steps")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.metric == "success_rate" and not is_metaworld(args.env):
        p.error("success_rate is only recorded for MetaWorld tasks")
    bin_size = args.bin_size or (10000 if args.axis == "env_steps" else 100)
    if bin_size <= 0:
        p.error("--bin-size must be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    overlay = None
    if args.metric in ("reward", "gt_reward", "learned_reward"):
        overlay = OVERLAY_GROUPS[0]
    elif args.metric in ("return", "gt_return_mean", "learned_return_mean"):
        overlay = OVERLAY_GROUPS[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    if overlay:
        source_name, (gt_metric, learned_metric), stem, gt_label, learned_label = overlay
        gt_runs, learned_runs = [], []
        for mode in ("certain", "probabilistic"):
            for budget in (1000, 5000, 10000):
                files = sorted((args.root / args.env / mode / f"feedback_{budget}").glob(
                    f"seed_*/{source_name}"))
                if not files:
                    continue
                gts, learneds = [], []
                for file in files:
                    config_path = file.parent / "config.json"
                    config = (json.loads(config_path.read_text(encoding="utf-8"))
                              if config_path.exists() else {})
                    gts.append(load_series(file, gt_metric, args.axis, bin_size,
                                           _cutoff(config, gt_metric, args.axis)))
                    learneds.append(load_series(file, learned_metric, args.axis, bin_size,
                                                _cutoff(config, learned_metric, args.axis)))
                gt_common = sorted(set.intersection(*(set(run) for run in gts)))
                learned_common = sorted(set.intersection(*(set(run) for run in learneds)))
                label = f"{mode}, {budget} feedback (n={len(files)})"
                if gt_common:
                    x = np.asarray(gt_common)
                    y = np.asarray([[run[point] for point in gt_common] for run in gts])
                    ax.plot(x, np.mean(y, axis=0), label=f"{label} {gt_label}")
                    found = True
                if learned_common:
                    x = np.asarray(learned_common)
                    y = np.asarray([[run[point] for point in learned_common] for run in learneds])
                    ax.plot(x, np.mean(y, axis=0), linestyle="--",
                            label=f"{label} {learned_label}")
                    found = True
        ax.set(xlabel=args.axis.replace("_", " "), ylabel="reward",
               title=f"{args.env}: ground truth and learned {stem}")
        if not found:
            raise SystemExit(f"No usable {source_name} files found for {args.env} under {args.root}")
    else:
        source = "evaluation.csv"
        for mode in ("certain", "probabilistic"):
            for budget in (1000, 5000, 10000):
                files = sorted((args.root / args.env / mode / f"feedback_{budget}").glob(
                    f"seed_*/{source}"))
                if not files:
                    continue
                runs = []
                for file in files:
                    config_path = file.parent / "config.json"
                    config = (json.loads(config_path.read_text(encoding="utf-8"))
                              if config_path.exists() else {})
                    runs.append(load_series(file, args.metric, args.axis, bin_size,
                                            _cutoff(config, args.metric, args.axis)))
                common = sorted(set.intersection(*(set(run) for run in runs)))
                if not common:
                    continue
                x = np.asarray(common)
                y = np.asarray([[run[point] for point in common] for run in runs])
                mean = np.mean(y, axis=0)
                std = np.std(y, axis=0)
                label = f"{mode}, {budget} feedback (n={len(runs)})"
                ax.plot(x, mean, label=label)
                if len(runs) > 1:
                    ax.fill_between(x, mean - std, mean + std, alpha=0.15)
                found = True
        if not found:
            raise SystemExit(f"No usable {source} files found for {args.env} under {args.root}")
        ax.set(xlabel=args.axis.replace("_", " "), ylabel=args.metric.replace("_", " "),
               title=f"{args.env}: {args.metric.replace('_', ' ')}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
