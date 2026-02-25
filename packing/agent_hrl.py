from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .replay import HierarchicalReplay, QTransition, ReplayBuffer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_episode(ep: int, start: float, end: float, decay: float) -> float:
    return float(end + (start - end) * np.exp(-ep / max(1e-6, decay)))


def argmax_q_index(q_net, map_states: np.ndarray, scalar_states: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        map_tensor = torch.from_numpy(map_states.astype(np.float32)).to(device)
        scalar_tensor = torch.from_numpy(scalar_states.astype(np.float32)).to(device)
        q_vals = q_net(map_tensor, scalar_tensor).detach().cpu().numpy()
    return int(np.argmax(q_vals))


def max_q(q_net, map_states: np.ndarray, scalar_states: np.ndarray, device: torch.device) -> float:
    if map_states.shape[0] == 0:
        return 0.0
    with torch.no_grad():
        map_tensor = torch.from_numpy(map_states.astype(np.float32)).to(device)
        scalar_tensor = torch.from_numpy(scalar_states.astype(np.float32)).to(device)
        return float(torch.max(q_net(map_tensor, scalar_tensor)).item())


def optimize_step(
    q_net,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float | None:
    if len(replay) < batch_size:
        return None

    maps, scalars, rewards, next_max_q, done = replay.sample_tensors(batch_size=batch_size, device=device)

    pred = q_net(maps, scalars)
    target = rewards + (1.0 - done) * gamma * next_max_q
    loss = F.smooth_l1_loss(pred, target.detach())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.item())


class HRLPackingAgent:
    """Thin orchestration wrapper for hierarchical Q-learning.

    This keeps action selection, replay management, and optimization in one place,
    while reusing the standalone helpers in this module.
    """

    def __init__(
        self,
        manager_q,
        manager_t,
        manager_opt: torch.optim.Optimizer,
        worker_q,
        worker_t,
        worker_opt: torch.optim.Optimizer,
        replay: HierarchicalReplay,
        gamma: float,
        target_update: int,
        device: torch.device,
    ) -> None:
        self.manager_q = manager_q
        self.manager_t = manager_t
        self.manager_opt = manager_opt
        self.worker_q = worker_q
        self.worker_t = worker_t
        self.worker_opt = worker_opt
        self.replay = replay
        self.gamma = float(gamma)
        self.target_update = int(target_update)
        self.device = device
        self.global_step = 0

    def act_manager(self, maps: np.ndarray, scalars: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if maps.shape[0] == 0:
            raise ValueError("manager candidate set is empty.")
        if rng.random() < epsilon:
            return int(rng.integers(0, maps.shape[0]))
        return argmax_q_index(self.manager_q, maps, scalars, self.device)

    def act_worker(self, maps: np.ndarray, scalars: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if maps.shape[0] == 0:
            raise ValueError("worker candidate set is empty.")
        if rng.random() < epsilon:
            return int(rng.integers(0, maps.shape[0]))
        return argmax_q_index(self.worker_q, maps, scalars, self.device)

    def remember_manager(self, transition: QTransition) -> None:
        self.replay.manager.add(transition)

    def remember_worker(self, transition: QTransition) -> None:
        self.replay.worker.add(transition)

    def train_step(self, batch_size: int) -> tuple[float | None, float | None]:
        manager_loss = optimize_step(
            q_net=self.manager_q,
            optimizer=self.manager_opt,
            replay=self.replay.manager,
            batch_size=batch_size,
            gamma=self.gamma,
            device=self.device,
        )
        worker_loss = optimize_step(
            q_net=self.worker_q,
            optimizer=self.worker_opt,
            replay=self.replay.worker,
            batch_size=batch_size,
            gamma=self.gamma,
            device=self.device,
        )
        self.global_step += 1
        if self.global_step % self.target_update == 0:
            self.manager_t.load_state_dict(self.manager_q.state_dict())
            self.worker_t.load_state_dict(self.worker_q.state_dict())
        return manager_loss, worker_loss

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "manager_state_dict": self.manager_q.state_dict(),
            "worker_state_dict": self.worker_q.state_dict(),
            "global_step": self.global_step,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.manager_q.load_state_dict(ckpt["manager_state_dict"], strict=strict)
        self.worker_q.load_state_dict(ckpt["worker_state_dict"], strict=strict)
        self.manager_t.load_state_dict(self.manager_q.state_dict())
        self.worker_t.load_state_dict(self.worker_q.state_dict())
        self.global_step = int(ckpt.get("global_step", 0))
        return ckpt
