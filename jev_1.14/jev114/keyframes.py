"""Compress a long segment to a fixed-length clip of salient keyframes.

Candidates come from three complementary cues, then a short DP picks exactly
``n_out`` strictly increasing indices (always including the first and last step):

* near-zero velocity (pauses / holds)
* peaks of residual vs a moving-average smoother (sharp motion)
* ruptures change-point detection (mean shifts); numpy binary segmentation if
  ruptures is not installed
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .learning import Segment

try:
    import ruptures as rpt
except ImportError:
    rpt = None


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError("trajectory arrays must be 1-D or 2-D")
    return x


def _zscore(x: np.ndarray) -> np.ndarray:
    scale = np.std(x, axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - np.mean(x, axis=0, keepdims=True)) / scale


def _normalize(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    lo, hi = float(score.min()), float(score.max())
    if hi - lo < 1e-12:
        return np.zeros_like(score)
    return (score - lo) / (hi - lo)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(x)))
    if window == 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(x, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, j], kernel, mode="valid")[:len(x)]
                       for j in range(x.shape[1])], axis=1)
    return smooth


def velocity_norm(obs: np.ndarray) -> np.ndarray:
    obs = _as_2d(obs)
    delta = np.diff(obs, axis=0)
    speed = np.linalg.norm(delta, axis=1)
    return np.concatenate([[speed[0] if len(speed) else 0.0], speed])


def near_zero_velocity_score(obs: np.ndarray) -> np.ndarray:
    """High where the observation trajectory is almost still."""
    speed = velocity_norm(obs)
    inverted = 1.0 / (speed + 1e-6)
    score = _normalize(inverted)
    if len(speed) >= 3:
        minima = np.zeros_like(score)
        interior = (speed[1:-1] <= speed[:-2]) & (speed[1:-1] <= speed[2:])
        minima[1:-1] = interior.astype(np.float64)
        score = score + 0.5 * minima
    quiet = speed <= np.quantile(speed, 0.2)
    score = score + 0.5 * quiet.astype(np.float64)
    return score


def smoothing_residual_score(obs: np.ndarray, window: int = 5) -> np.ndarray:
    """High where the trajectory departs from a local smoother (corners, jerks)."""
    obs = _as_2d(obs)
    residual = np.linalg.norm(obs - _moving_average(obs, window), axis=1)
    score = _normalize(residual)
    if len(residual) >= 3:
        peaks = np.zeros_like(score)
        interior = (residual[1:-1] >= residual[:-2]) & (residual[1:-1] >= residual[2:])
        peaks[1:-1] = interior.astype(np.float64)
        score = score + 0.5 * peaks
    return score


def _binary_segmentation_bkps(signal: np.ndarray, n_bkps: int) -> list[int]:
    """Greedy mean-shift splits; used when ruptures is unavailable."""
    n, _ = signal.shape
    bounds = [0, n]
    for _ in range(max(0, n_bkps)):
        best_gain, best_t, best_i = 0.0, None, None
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            if end - start < 4:
                continue
            seg = signal[start:end]
            total = np.mean(seg, axis=0)
            for t in range(start + 2, end - 1):
                left = np.mean(signal[start:t], axis=0)
                right = np.mean(signal[t:end], axis=0)
                gain = (t - start) * float(np.sum((left - total) ** 2))
                gain += (end - t) * float(np.sum((right - total) ** 2))
                if gain > best_gain:
                    best_gain, best_t, best_i = gain, t, i
        if best_t is None:
            break
        bounds.insert(best_i + 1, best_t)
    return bounds[1:]


def changepoint_indices(obs: np.ndarray, act: np.ndarray | None = None,
                        n_bkps: int | None = None) -> np.ndarray:
    """Segment end-indices (last value is T), from ruptures PELT/Binseg or a fallback."""
    signal = _zscore(_as_2d(obs))
    if act is not None:
        signal = np.concatenate([signal, _zscore(_as_2d(act))], axis=1)
    t = signal.shape[0]
    if t < 4:
        return np.array([t], dtype=np.int64)
    n_bkps = int(n_bkps) if n_bkps is not None else max(2, min(12, t // 4))
    n_bkps = max(1, min(n_bkps, t - 2))
    if rpt is not None:
        algo = rpt.Binseg(model="l2", min_size=2, jump=1).fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)
        return np.asarray(bkps, dtype=np.int64)
    return np.asarray(_binary_segmentation_bkps(signal, n_bkps), dtype=np.int64)


def changepoint_score(obs: np.ndarray, act: np.ndarray | None = None,
                      n_bkps: int | None = None) -> np.ndarray:
    t = len(obs)
    score = np.zeros(t, dtype=np.float64)
    bkps = changepoint_indices(obs, act, n_bkps=n_bkps)
    for end in bkps:
        # ruptures reports exclusive segment ends; the change lives at end-1.
        i = int(end) - 1
        if 0 < i < t:
            score[i] = 1.0
    return score


def combined_keyframe_score(obs: np.ndarray, act: np.ndarray | None = None,
                            smooth_window: int = 5,
                            n_bkps: int | None = None) -> np.ndarray:
    return (_normalize(near_zero_velocity_score(obs))
            + _normalize(smoothing_residual_score(obs, window=smooth_window))
            + changepoint_score(obs, act, n_bkps=n_bkps))


def select_keyframe_indices(score: np.ndarray, n_out: int) -> np.ndarray:
    """Pick ``n_out`` increasing times, forcing the first and last frames."""
    score = np.asarray(score, dtype=np.float64)
    t = int(score.shape[0])
    n_out = int(n_out)
    if n_out <= 0:
        raise ValueError("n_out must be positive")
    if t <= n_out:
        return np.arange(t, dtype=np.int64)
    if n_out == 1:
        return np.array([0], dtype=np.int64)
    neg = -1e18
    dp = np.full((n_out, t), neg, dtype=np.float64)
    prev = np.full((n_out, t), -1, dtype=np.int64)
    dp[0, 0] = score[0]
    for k in range(1, n_out):
        running, running_s = neg, -1
        for i in range(k, t):
            s = i - 1
            if dp[k - 1, s] > running:
                running, running_s = dp[k - 1, s], s
            dp[k, i] = score[i] + running
            prev[k, i] = running_s
    if dp[n_out - 1, t - 1] <= neg / 2:
        return np.unique(np.linspace(0, t - 1, n_out).round().astype(np.int64))
    idx = t - 1
    chosen = np.empty(n_out, dtype=np.int64)
    for k in range(n_out - 1, -1, -1):
        chosen[k] = idx
        idx = int(prev[k, idx])
    chosen[0] = 0
    chosen[-1] = t - 1
    return chosen


def keyframe_indices(obs: np.ndarray, n_out: int, act: np.ndarray | None = None,
                     smooth_window: int = 5, n_bkps: int | None = None) -> np.ndarray:
    score = combined_keyframe_score(obs, act, smooth_window=smooth_window, n_bkps=n_bkps)
    return select_keyframe_indices(score, n_out)


def subsample_segment(segment: Segment, n_out: int, **kwargs) -> tuple[Segment, np.ndarray]:
    """Return a length-``n_out`` segment and the source indices in the original clip."""
    idx = keyframe_indices(segment.obs, n_out, act=segment.act, **kwargs)
    states = None
    if segment.states is not None:
        states = tuple(segment.states[int(i)] for i in idx)
    clipped = replace(segment, obs=np.asarray(segment.obs)[idx],
                      act=np.asarray(segment.act)[idx], states=states)
    return clipped, idx
