from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class QTransition:
    map_state: np.ndarray | torch.Tensor
    scalar_state: np.ndarray | torch.Tensor
    reward: float
    next_max_q: float
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        storage_device: torch.device | str | None = None,
        allow_cpu_fallback_on_oom: bool = False,
    ):
        self.capacity = int(capacity)
        self.storage_device = torch.device(storage_device) if storage_device is not None else None
        self.allow_cpu_fallback_on_oom = bool(allow_cpu_fallback_on_oom)
        self.buffer: deque[QTransition] = deque(maxlen=capacity)
        self._warned_fallback = False
        self._maps_t: torch.Tensor | None = None
        self._scalars_t: torch.Tensor | None = None
        self._rewards_t: torch.Tensor | None = None
        self._next_max_q_t: torch.Tensor | None = None
        self._done_t: torch.Tensor | None = None
        self._size = 0
        self._write_idx = 0

    def _as_float_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.storage_device, dtype=torch.float32)
        arr = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(arr).to(device=self.storage_device, dtype=torch.float32)

    def _lazy_init_storage(self, transition: QTransition) -> None:
        if self._maps_t is not None:
            return
        map_t = self._as_float_tensor(transition.map_state)
        scalar_t = self._as_float_tensor(transition.scalar_state)
        self._maps_t = torch.empty((self.capacity, *map_t.shape), dtype=torch.float32, device=self.storage_device)
        self._scalars_t = torch.empty((self.capacity, *scalar_t.shape), dtype=torch.float32, device=self.storage_device)
        self._rewards_t = torch.empty((self.capacity,), dtype=torch.float32, device=self.storage_device)
        self._next_max_q_t = torch.empty((self.capacity,), dtype=torch.float32, device=self.storage_device)
        self._done_t = torch.empty((self.capacity,), dtype=torch.float32, device=self.storage_device)

    def _fallback_to_cpu_storage(self) -> None:
        if not self._warned_fallback:
            print("note=replay_gpu_oom_fallback using CPU replay storage")
            self._warned_fallback = True
        self.storage_device = None
        self._maps_t = None
        self._scalars_t = None
        self._rewards_t = None
        self._next_max_q_t = None
        self._done_t = None
        self._size = 0
        self._write_idx = 0
        self.buffer = deque(maxlen=self.capacity)

    def add(self, transition: QTransition) -> None:
        if self.storage_device is not None:
            try:
                self._lazy_init_storage(transition)
                assert self._maps_t is not None
                assert self._scalars_t is not None
                assert self._rewards_t is not None
                assert self._next_max_q_t is not None
                assert self._done_t is not None
                idx = self._write_idx
                self._maps_t[idx].copy_(self._as_float_tensor(transition.map_state))
                self._scalars_t[idx].copy_(self._as_float_tensor(transition.scalar_state))
                self._rewards_t[idx] = float(transition.reward)
                self._next_max_q_t[idx] = float(transition.next_max_q)
                self._done_t[idx] = 1.0 if bool(transition.done) else 0.0
                self._write_idx = (self._write_idx + 1) % self.capacity
                self._size = min(self.capacity, self._size + 1)
                return
            except torch.OutOfMemoryError:
                if self.allow_cpu_fallback_on_oom:
                    # Graceful degradation: keep training on CPU replay when GPU replay is too large.
                    self._fallback_to_cpu_storage()
                else:
                    raise RuntimeError(
                        "GPU replay allocation failed (OOM). "
                        "Reduce --replay_size or use a larger GPU. "
                        "If you want automatic fallback, construct ReplayBuffer with "
                        "allow_cpu_fallback_on_oom=True."
                    )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[QTransition]:
        if self.storage_device is not None:
            if self._size < batch_size:
                raise ValueError("Not enough transitions in replay buffer.")
            idx = np.random.choice(self._size, size=batch_size, replace=False)
            assert self._maps_t is not None
            assert self._scalars_t is not None
            assert self._rewards_t is not None
            assert self._next_max_q_t is not None
            assert self._done_t is not None
            out: list[QTransition] = []
            for i in idx:
                out.append(
                    QTransition(
                        map_state=self._maps_t[i].detach().cpu().numpy(),
                        scalar_state=self._scalars_t[i].detach().cpu().numpy(),
                        reward=float(self._rewards_t[i].item()),
                        next_max_q=float(self._next_max_q_t[i].item()),
                        done=bool(self._done_t[i].item() > 0.5),
                    )
                )
            return out
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def sample_tensors(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.storage_device is not None:
            if self._size < batch_size:
                raise ValueError("Not enough transitions in replay buffer.")
            assert self._maps_t is not None
            assert self._scalars_t is not None
            assert self._rewards_t is not None
            assert self._next_max_q_t is not None
            assert self._done_t is not None
            # Use randint instead of randperm for better performance with large replay sizes
            idx_t = torch.randint(0, self._size, (batch_size,), device=self.storage_device)
            maps = self._maps_t.index_select(0, idx_t)
            scalars = self._scalars_t.index_select(0, idx_t)
            rewards = self._rewards_t.index_select(0, idx_t)
            next_max_q = self._next_max_q_t.index_select(0, idx_t)
            done = self._done_t.index_select(0, idx_t)
            if device != self.storage_device:
                maps = maps.to(device=device, non_blocking=True)
                scalars = scalars.to(device=device, non_blocking=True)
                rewards = rewards.to(device=device, non_blocking=True)
                next_max_q = next_max_q.to(device=device, non_blocking=True)
                done = done.to(device=device, non_blocking=True)
            return maps, scalars, rewards, next_max_q, done

        batch = self.sample(batch_size)
        
        def _to_numpy(x) -> np.ndarray:
            """Convert tensor or array to numpy, handling CUDA tensors."""
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.asarray(x, dtype=np.float32)
        
        maps = torch.from_numpy(np.stack([_to_numpy(t.map_state) for t in batch], axis=0)).to(device)
        scalars = torch.from_numpy(np.stack([_to_numpy(t.scalar_state) for t in batch], axis=0)).to(device)
        rewards = torch.tensor([float(t.reward) for t in batch], dtype=torch.float32, device=device)
        next_max_q = torch.tensor([float(t.next_max_q) for t in batch], dtype=torch.float32, device=device)
        done = torch.tensor([1.0 if t.done else 0.0 for t in batch], dtype=torch.float32, device=device)
        return maps, scalars, rewards, next_max_q, done

    def __len__(self) -> int:
        if self.storage_device is not None:
            return self._size
        return len(self.buffer)

    def state_dict(self) -> dict:
        """Return a CPU-serialisable snapshot of the replay buffer."""
        if self.storage_device is not None and self._maps_t is not None:
            return {
                "mode": "tensor",
                "capacity": self.capacity,
                "size": self._size,
                "write_idx": self._write_idx,
                "maps": self._maps_t[:self._size].detach().cpu(),
                "scalars": self._scalars_t[:self._size].detach().cpu(),
                "rewards": self._rewards_t[:self._size].detach().cpu(),
                "next_max_q": self._next_max_q_t[:self._size].detach().cpu(),
                "done": self._done_t[:self._size].detach().cpu(),
            }
        # deque / uninitialized tensor path
        transitions = list(self.buffer)
        return {
            "mode": "deque",
            "capacity": self.capacity,
            "maps": [np.asarray(t.map_state, dtype=np.float32) for t in transitions],
            "scalars": [np.asarray(t.scalar_state, dtype=np.float32) for t in transitions],
            "rewards": [float(t.reward) for t in transitions],
            "next_max_q": [float(t.next_max_q) for t in transitions],
            "done": [bool(t.done) for t in transitions],
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore replay buffer from a state_dict snapshot."""
        self.capacity = int(sd["capacity"])
        if sd["mode"] == "tensor" and self.storage_device is not None:
            n = int(sd["size"])
            if n == 0:
                return
            # Re-init storage with correct shapes
            maps_cpu: torch.Tensor = sd["maps"]
            scalars_cpu: torch.Tensor = sd["scalars"]
            self._maps_t = torch.empty((self.capacity, *maps_cpu.shape[1:]), dtype=torch.float32, device=self.storage_device)
            self._scalars_t = torch.empty((self.capacity, *scalars_cpu.shape[1:]), dtype=torch.float32, device=self.storage_device)
            self._rewards_t = torch.empty((self.capacity,), dtype=torch.float32, device=self.storage_device)
            self._next_max_q_t = torch.empty((self.capacity,), dtype=torch.float32, device=self.storage_device)
            self._done_t = torch.empty((self.capacity,), dtype=torch.float32, device=self.storage_device)
            self._maps_t[:n].copy_(maps_cpu)
            self._scalars_t[:n].copy_(scalars_cpu)
            self._rewards_t[:n].copy_(sd["rewards"])
            self._next_max_q_t[:n].copy_(sd["next_max_q"])
            self._done_t[:n].copy_(sd["done"])
            self._size = n
            self._write_idx = int(sd.get("write_idx", n % self.capacity))
        else:
            # Restore as deque
            self.buffer = deque(maxlen=self.capacity)
            maps_list = sd["maps"]
            for i in range(len(maps_list)):
                self.buffer.append(QTransition(
                    map_state=np.asarray(maps_list[i], dtype=np.float32),
                    scalar_state=np.asarray(sd["scalars"][i], dtype=np.float32),
                    reward=float(sd["rewards"][i]),
                    next_max_q=float(sd["next_max_q"][i]),
                    done=bool(sd["done"][i]),
                ))


@dataclass
class HierarchicalReplay:
    manager: ReplayBuffer
    worker: ReplayBuffer
