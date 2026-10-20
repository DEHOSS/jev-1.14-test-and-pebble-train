"""SAC, replay, and an ensemble Bradley-Terry reward model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .sampling import (disagreement_threshold, pair_indistinguishable, sepoa_scores,
                       skill_total_variation)


def mlp(input_dim: int, output_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))


def skill_one_hot(skill_ids: np.ndarray, n_skills: int) -> np.ndarray:
    ids = np.atleast_1d(np.asarray(skill_ids, dtype=np.int64))
    out = np.zeros((ids.shape[0], n_skills), dtype=np.float32)
    np.put_along_axis(out, ids[:, None], 1.0, axis=-1)
    return out


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.act = np.empty((capacity, act_dim), np.float32)
        self.gt = np.empty(capacity, np.float32)
        self.learned = np.empty(capacity, np.float32)
        self.terminal = np.empty(capacity, np.float32)
        self.skill = np.empty(capacity, np.int64)

    def add(self, obs, act, gt, next_obs, terminal, learned=np.nan, skill=0):
        i = self.pos
        self.obs[i], self.act[i], self.gt[i] = obs, act, gt
        self.next_obs[i], self.terminal[i], self.learned[i] = next_obs, terminal, learned
        self.skill[i] = int(skill)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: str):
        idx = rng.integers(self.size, size=batch_size)
        tensors = tuple(torch.as_tensor(x[idx], device=device) for x in
                        (self.obs, self.act, self.learned, self.next_obs, self.terminal))
        skills = torch.as_tensor(self.skill[idx], device=device, dtype=torch.long)
        return tensors + (skills,)

    def neighbor_obs(self, rng: np.random.Generator, limit: int = 4096) -> np.ndarray:
        count = min(self.size, limit)
        if count == self.size:
            return self.obs[:self.size]
        idx = rng.choice(self.size, size=count, replace=False)
        return self.obs[idx]

    def relabel(self, model: "RewardEnsemble", chunk: int = 8192):
        for start in range(0, self.size, chunk):
            end = min(start + chunk, self.size)
            self.learned[start:end] = model.predict(self.obs[start:end], self.act[start:end])


def knn_entropy_rewards(query: np.ndarray, memory: np.ndarray, k: int = 5) -> np.ndarray:
    """PEBBLE/APT-style intrinsic reward: log distance to the k-th nearest state."""
    if len(query) == 0 or len(memory) == 0:
        return np.zeros(len(query), np.float32)
    k = max(1, min(k, len(memory)))
    delta = query[:, None, :] - memory[None, :, :]
    dist = np.sqrt(np.sum(np.square(delta), axis=-1) + 1e-8)
    knn = np.partition(dist, k - 1, axis=1)[:, k - 1]
    return np.log(knn + 1.0).astype(np.float32)


@dataclass
class Segment:
    obs: np.ndarray
    act: np.ndarray
    gt_return: float
    skill_id: int = 0
    states: tuple | None = None


class EpisodeStore:
    def __init__(self, segment_length: int, max_episodes: int = 1000):
        self.segment_length = segment_length
        self.episodes: deque[tuple] = deque(maxlen=max_episodes)
        self._obs: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._gt: list[float] = []
        self._skill: list[int] = []
        self._state: list = []

    def add(self, obs, act, gt: float, done: bool, skill: int = 0, state=None):
        self._obs.append(np.array(obs, copy=True))
        self._act.append(np.array(act, copy=True))
        self._gt.append(gt)
        self._skill.append(int(skill))
        self._state.append(state)
        if done:
            if len(self._obs) >= self.segment_length:
                states = tuple(self._state)
                if any(item is None for item in states):
                    states = None
                self.episodes.append((np.stack(self._obs), np.stack(self._act),
                                      np.asarray(self._gt, dtype=np.float32),
                                      np.asarray(self._skill, dtype=np.int64),
                                      states))
            self._obs, self._act, self._gt, self._skill, self._state = [], [], [], [], []

    def eligible_episodes(self, length: int | None = None):
        length = self.segment_length if length is None else int(length)
        return [ep for ep in self.episodes if len(ep[2]) >= length]

    def sample(self, rng: np.random.Generator, length: int | None = None) -> Segment:
        length = self.segment_length if length is None else int(length)
        eligible = self.eligible_episodes(length)
        if not eligible:
            raise ValueError("No complete episode long enough for a feedback segment")
        obs, act, gt, skill, states = eligible[int(rng.integers(len(eligible)))]
        start = int(rng.integers(len(gt) - length + 1))
        sl = slice(start, start + length)
        skill_id = int(np.bincount(skill[sl]).argmax())
        window = None if states is None else states[sl]
        return Segment(obs[sl].copy(), act[sl].copy(), float(gt[sl].sum()), skill_id,
                       window)


@dataclass
class Preference:
    a: Segment
    b: Segment
    target_b: float


class RewardEnsemble:
    def __init__(self, obs_dim: int, act_dim: int, device: str, members: int = 3,
                 hidden: int = 256, lr: float = 3e-4):
        self.device = device
        self.members = nn.ModuleList([mlp(obs_dim + act_dim, 1, hidden).to(device)
                                      for _ in range(members)])
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=lr) for net in self.members]
        self.fitted = False

    @torch.no_grad()
    def predict(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        if len(obs) == 0:
            return np.empty(0, dtype=np.float32)
        x = torch.as_tensor(np.concatenate((obs, act), axis=-1), dtype=torch.float32,
                            device=self.device)
        values = torch.stack([net(x).squeeze(-1) for net in self.members]).mean(0)
        return values.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def disagreement(self, pairs: list[tuple[Segment, Segment]]) -> np.ndarray:
        values = []
        for a, b in pairs:
            x_a = torch.as_tensor(np.concatenate((a.obs, a.act), -1), device=self.device)
            x_b = torch.as_tensor(np.concatenate((b.obs, b.act), -1), device=self.device)
            logits = torch.stack([(net(x_b).sum() - net(x_a).sum()) for net in self.members])
            values.append(torch.sigmoid(logits).std(unbiased=False).item())
        return np.asarray(values)

    def segment_return(self, segment: Segment) -> float:
        return float(self.predict(segment.obs, segment.act).sum())

    def _preference_tensors(self, data: list[Preference]):
        a = torch.as_tensor(np.stack([np.concatenate((p.a.obs, p.a.act), -1)
                                      for p in data]), device=self.device)
        b = torch.as_tensor(np.stack([np.concatenate((p.b.obs, p.b.act), -1)
                                      for p in data]), device=self.device)
        target = torch.as_tensor([p.target_b for p in data], dtype=torch.float32,
                                 device=self.device)
        return a, b, target

    def _ensemble_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(b).sum(dim=1).squeeze(-1) - net(a).sum(dim=1).squeeze(-1)
                            for net in self.members]).mean(0)

    @torch.no_grad()
    def preference_metrics(self, data: list[Preference]) -> tuple[float, float]:
        if not data:
            return float("nan"), float("nan")
        a, b, target = self._preference_tensors(data)
        logits = self._ensemble_logits(a, b)
        loss = float(F.binary_cross_entropy_with_logits(logits, target).item())
        decided = target != 0.5
        if not bool(decided.any()):
            return loss, float("nan")
        predicted_b = torch.sigmoid(logits[decided]) > 0.5
        labeled_b = target[decided] > 0.5
        accuracy = float((predicted_b == labeled_b).float().mean().item())
        return loss, accuracy

    def fit(self, data: list[Preference], rng: np.random.Generator, steps: int = 100,
            batch_size: int = 32) -> tuple[float, float]:
        if not data:
            return float("nan"), float("nan")
        losses = []
        for _ in range(steps):
            samples = [data[i] for i in rng.integers(len(data), size=batch_size)]
            a, b, target = self._preference_tensors(samples)
            for net, optimizer in zip(self.members, self.optimizers):
                logits = net(b).sum(dim=1).squeeze(-1) - net(a).sum(dim=1).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        self.fitted = True
        _, accuracy = self.preference_metrics(data)
        return float(np.mean(losses)), accuracy


class SkillDetector:
    """Maps a segment to a discrete skill posterior for S-EPOA query selection.

    ``diayn`` trains q(z|s) with cross-entropy (one-hot skills).
    ``aps`` trains successor features φ(s) with the URLB APS MLE loss
    −w⊤φ(s′) plus an APT kNN entropy bonus; the codebook of unit vectors w
    is the discrete skill set used at query time.
    """

    def __init__(self, obs_dim: int, n_skills: int, device: str, hidden: int = 256,
                 lr: float = 3e-4, method: str = "aps", sf_dim: int = 16,
                 knn_k: int = 12, posterior_temp: float = 5.0):
        if method not in ("aps", "diayn"):
            raise ValueError("skill method must be 'aps' or 'diayn'")
        self.n_skills, self.device, self.method = n_skills, device, method
        self.sf_dim = int(sf_dim) if method == "aps" else n_skills
        self.knn_k = max(1, int(knn_k))
        self.posterior_temp = float(posterior_temp)
        self.skill_dim = self.sf_dim if method == "aps" else n_skills
        out_dim = self.sf_dim if method == "aps" else n_skills
        self.net = mlp(obs_dim, out_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        if method == "aps":
            codebook = torch.randn(n_skills, self.sf_dim, device=device)
            self.codebook = F.normalize(codebook, dim=-1)
        else:
            self.codebook = None
        self.fitted = False

    def skill_vector(self, skill_id: int) -> np.ndarray:
        if self.method == "diayn":
            return skill_one_hot(np.array([skill_id]), self.n_skills)[0]
        return self.codebook[int(skill_id)].detach().cpu().numpy().astype(np.float32)

    def skill_tensor(self, skill_ids: torch.Tensor) -> torch.Tensor:
        ids = skill_ids.long().view(-1)
        if self.method == "diayn":
            return F.one_hot(ids, self.n_skills).float()
        return self.codebook[ids]

    def features(self, obs: torch.Tensor, norm: bool = True) -> torch.Tensor:
        feat = self.net(obs)
        if self.method == "aps" and norm:
            return F.normalize(feat, dim=-1)
        return feat

    def update(self, obs: torch.Tensor, skill_ids: torch.Tensor,
               next_obs: torch.Tensor | None = None) -> float:
        if self.method == "aps":
            states = obs if next_obs is None else next_obs
            phi = self.features(states, norm=True)
            loss = -(self.skill_tensor(skill_ids) * phi).sum(-1).mean()
        else:
            loss = F.cross_entropy(self.net(obs), skill_ids)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.fitted = True
        return float(loss.item())

    @torch.no_grad()
    def posterior(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) == 0:
            return np.full(self.n_skills, 1.0 / self.n_skills, np.float64)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.method == "diayn":
            probs = torch.softmax(self.net(x), dim=-1).mean(0)
            return probs.cpu().numpy().astype(np.float64)
        phi = self.features(x, norm=True).mean(0)
        logits = self.posterior_temp * (self.codebook @ phi)
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)

    def detect(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.posterior(obs)))

    @torch.no_grad()
    def diayn_rewards(self, obs: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        log_q = F.log_softmax(self.net(obs), dim=-1)
        prior = float(np.log(self.n_skills))
        return log_q.gather(1, skill_ids.unsqueeze(-1)).squeeze(-1) + prior

    @torch.no_grad()
    def aps_rewards(self, next_obs: torch.Tensor, skill_ids: torch.Tensor,
                    memory_obs: torch.Tensor) -> torch.Tensor:
        """URLB APS: APT particle entropy of φ(s′) plus successor-feature term w⊤φ(s′)."""
        phi_query = self.features(next_obs, norm=False)
        phi_memory = self.features(memory_obs, norm=False)
        entropy = knn_entropy_rewards(phi_query.cpu().numpy(), phi_memory.cpu().numpy(),
                                      k=self.knn_k)
        entropy = torch.as_tensor(entropy, device=self.device, dtype=torch.float32)
        sf = (self.skill_tensor(skill_ids) * F.normalize(phi_query, dim=-1)).sum(-1)
        return entropy + sf


class SkillReturnEstimator:
    """R_θ(z): expected learned return of trajectories generated by skill z."""

    def __init__(self, n_skills: int, device: str, hidden: int = 64, lr: float = 3e-4):
        self.n_skills, self.device = n_skills, device
        self.net = mlp(n_skills, 1, hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.fitted = False

    def update(self, skill_ids: np.ndarray, returns: np.ndarray) -> float:
        skill_ids = np.asarray(skill_ids)
        returns = np.asarray(returns, dtype=np.float64)
        ok = np.isfinite(returns)
        if not np.any(ok):
            return float("nan")
        skill_ids, returns = skill_ids[ok], returns[ok]
        z = torch.as_tensor(skill_one_hot(skill_ids, self.n_skills), device=self.device)
        target = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        lo, hi = target.min(), target.max()
        if float(hi - lo) > 1e-6:
            target = (target - lo) / (hi - lo)
        pred = self.net(z)
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.fitted = True
        return float(loss.item())

    @torch.no_grad()
    def predict_ids(self, skill_ids) -> np.ndarray:
        ids = np.atleast_1d(np.asarray(skill_ids, dtype=np.int64))
        z = torch.as_tensor(skill_one_hot(ids, self.n_skills), device=self.device)
        return self.net(z).squeeze(-1).cpu().numpy().astype(np.float64)

    def best_skill(self) -> int:
        if not self.fitted:
            return 0
        return int(np.argmax(self.predict_ids(np.arange(self.n_skills))))


def select_preference_pair(store: EpisodeStore, model: RewardEnsemble, rng: np.random.Generator,
                           candidates: int, attempts: int, disagreement_percentile: float,
                           disagreement_history: deque[float], detector: SkillDetector,
                           estimator: SkillReturnEstimator, min_skill_tv: float
                           ) -> tuple[Segment, Segment, dict]:
    """Reject indistinguishable pairs, then pick max S-EPOA I among the rest.

    A pair is indistinguishable if skill posteriors are too close (TV < min_skill_tv)
    or, after the reward ensemble is fitted, disagreement is below the adaptive
    percentile. Eligible pairs are ranked by I = (1+|ΔR_θ(z)|)(1+disagreement).
    """
    def draw_pairs():
        pairs = []
        for _ in range(max(1, candidates)):
            a, b = store.sample(rng), store.sample(rng)
            for _ in range(8):
                if not np.array_equal(a.obs, b.obs):
                    break
                b = store.sample(rng)
            pairs.append((a, b))
        return pairs

    def posts(pair):
        if not detector.fitted:
            n = detector.n_skills
            return np.full(n, 1.0 / n), np.full(n, 1.0 / n)
        return detector.posterior(pair[0].obs), detector.posterior(pair[1].obs)

    def skill_gap(pair, post_a, post_b):
        if estimator.fitted:
            za, zb = int(np.argmax(post_a)), int(np.argmax(post_b))
            returns = estimator.predict_ids([za, zb])
            return abs(float(returns[0] - returns[1]))
        return skill_total_variation(post_a, post_b)

    seen: list[tuple[Segment, Segment]] = []
    meta: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
    threshold = 0.0
    for _ in range(max(1, attempts)):
        batch = draw_pairs()
        seen.extend(batch)
        if not detector.fitted and not model.fitted:
            pair = batch[int(rng.integers(len(batch)))]
            return pair[0], pair[1], {"disagreement": float("nan"), "disagreement_threshold": float("nan"),
                                      "skill_tv": float("nan"), "skill_gap": float("nan"),
                                      "sepoa_score": float("nan"), "skill_a": -1, "skill_b": -1}
        disagreements = (np.asarray(model.disagreement(batch), dtype=np.float64)
                         if model.fitted else np.zeros(len(batch)))
        if model.fitted and disagreement_percentile > 0:
            for score in disagreements:
                disagreement_history.append(float(score))
            threshold = disagreement_threshold(np.asarray(disagreement_history),
                                               disagreement_percentile)
        posts_a, posts_b, gaps, tvs = [], [], [], []
        for pair in batch:
            pa, pb = posts(pair)
            posts_a.append(pa)
            posts_b.append(pb)
            tvs.append(skill_total_variation(pa, pb))
            gaps.append(skill_gap(pair, pa, pb))
        scores = sepoa_scores(np.asarray(gaps), disagreements)
        eligible = []
        for i, pair in enumerate(batch):
            meta.append((float(disagreements[i]), float(gaps[i]), float(tvs[i]),
                         posts_a[i], posts_b[i]))
            if pair_indistinguishable(posts_a[i], posts_b[i], float(disagreements[i]),
                                      min_skill_tv, threshold):
                continue
            eligible.append(i)
        if eligible:
            local = np.asarray([scores[i] for i in eligible])
            pick = eligible[int(np.argmax(local))]
            chosen = batch[pick]
            pa, pb = posts_a[pick], posts_b[pick]
            return chosen[0], chosen[1], {
                "disagreement": float(disagreements[pick]),
                "disagreement_threshold": threshold,
                "skill_tv": float(tvs[pick]),
                "skill_gap": float(gaps[pick]),
                "sepoa_score": float(scores[pick]),
                "skill_a": int(np.argmax(pa)),
                "skill_b": int(np.argmax(pb)),
            }
    if not meta:
        chosen = seen[0]
        return chosen[0], chosen[1], {"disagreement": float("nan"), "disagreement_threshold": threshold,
                                      "skill_tv": float("nan"), "skill_gap": float("nan"),
                                      "sepoa_score": float("nan"), "skill_a": -1, "skill_b": -1}
    gaps = np.array([m[1] for m in meta])
    diss = np.array([m[0] for m in meta])
    scores = sepoa_scores(gaps, diss)
    pick = int(np.argmax(scores))
    chosen = seen[pick]
    pa, pb = meta[pick][3], meta[pick][4]
    return chosen[0], chosen[1], {
        "disagreement": float(diss[pick]),
        "disagreement_threshold": threshold,
        "skill_tv": float(meta[pick][2]),
        "skill_gap": float(gaps[pick]),
        "sepoa_score": float(scores[pick]),
        "skill_a": int(np.argmax(pa)),
        "skill_b": int(np.argmax(pb)),
    }


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, skill_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim + skill_dim, 2 * act_dim, hidden)

    def sample(self, obs: torch.Tensor, skill: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.net(torch.cat((obs, skill), -1)).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        log_prob = dist.log_prob(raw) - 2 * (np.log(2) - raw - F.softplus(-2 * raw))
        return action, log_prob.sum(-1, keepdim=True)


class SuccessorQ(nn.Module):
    """APS critic: Q(s,a,w)=ψ(s,a)⊤w."""

    def __init__(self, obs_dim: int, act_dim: int, sf_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, sf_dim, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        psi = self.net(torch.cat((obs, act), -1))
        return (psi * skill).sum(-1, keepdim=True)


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, n_skills: int, device: str,
                 hidden: int = 256, lr: float = 3e-4, discount: float = 0.99, tau: float = 0.005,
                 skill_dim: int | None = None, successor_critic: bool = False):
        self.device, self.discount, self.tau = device, discount, tau
        self.n_skills = n_skills
        self.skill_dim = n_skills if skill_dim is None else int(skill_dim)
        self.successor_critic = successor_critic
        self.actor = Actor(obs_dim, act_dim, self.skill_dim, hidden).to(device)
        if successor_critic:
            self.q1 = SuccessorQ(obs_dim, act_dim, self.skill_dim, hidden).to(device)
            self.q2 = SuccessorQ(obs_dim, act_dim, self.skill_dim, hidden).to(device)
            self.tq1 = SuccessorQ(obs_dim, act_dim, self.skill_dim, hidden).to(device)
            self.tq2 = SuccessorQ(obs_dim, act_dim, self.skill_dim, hidden).to(device)
        else:
            self.q1, self.q2 = (mlp(obs_dim + act_dim, 1, hidden).to(device) for _ in range(2))
            self.tq1, self.tq2 = (mlp(obs_dim + act_dim, 1, hidden).to(device) for _ in range(2))
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)

    def _q(self, net, obs: torch.Tensor, act: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        if self.successor_critic:
            return net(obs, act, skill)
        return net(torch.cat((obs, act), -1))

    def _skill_tensor(self, skill, batch: int) -> torch.Tensor:
        if torch.is_tensor(skill) and skill.dim() == 2:
            return skill
        ids = torch.as_tensor(skill, device=self.device, dtype=torch.long).view(-1)
        if ids.numel() == 1 and batch > 1:
            ids = ids.expand(batch)
        return F.one_hot(ids, self.skill_dim).float()

    @torch.no_grad()
    def act(self, obs: np.ndarray, skill, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if np.ndim(skill) == 0:
            z = self._skill_tensor(int(skill), 1)
        else:
            z = torch.as_tensor(skill, dtype=torch.float32, device=self.device).view(1, -1)
        return self.actor.sample(x, z, deterministic)[0][0].cpu().numpy()

    def update(self, batch) -> dict[str, float]:
        obs, act, reward, next_obs, terminal, skill = batch
        reward, terminal = reward.unsqueeze(-1), terminal.unsqueeze(-1)
        skill = self._skill_tensor(skill, len(obs))
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs, skill)
            target_q = torch.min(self._q(self.tq1, next_obs, next_act, skill),
                                 self._q(self.tq2, next_obs, next_act, skill))
            backup = reward + self.discount * (1 - terminal) * (
                target_q - self.log_alpha.exp() * next_logp)
        q_loss = (F.mse_loss(self._q(self.q1, obs, act, skill), backup)
                  + F.mse_loss(self._q(self.q2, obs, act, skill), backup))
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        new_act, logp = self.actor.sample(obs, skill)
        q_new = torch.min(self._q(self.q1, obs, new_act, skill),
                          self._q(self.q2, obs, new_act, skill))
        actor_loss = (self.log_alpha.exp().detach() * logp - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for source, target in ((self.q1, self.tq1), (self.q2, self.tq2)):
                for p, tp in zip(source.parameters(), target.parameters()):
                    tp.lerp_(p, self.tau)
        return {"q_loss": float(q_loss.item()), "actor_loss": float(actor_loss.item()),
                "alpha": float(self.log_alpha.exp().item())}
