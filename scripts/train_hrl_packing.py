#!/usr/bin/env python

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from typing import Any

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from packing import PackingEnv
from packing.agent_hrl import argmax_q_index, epsilon_by_episode, max_q, optimize_step, set_seed
from packing.models_manager import ManagerQNet
from packing.models_worker import WorkerQNet
from packing.replay import HierarchicalReplay, QTransition, ReplayBuffer
from packing.reward import RewardWeights, transition_reward
from packing.state import (
    MANAGER_MAP_CHANNELS,
    MANAGER_SCALAR_DIM,
    PAPER_MANAGER_TOP_K,
    PAPER_ORIENTATION_STEP,
    WORKER_MAP_CHANNELS,
    WORKER_SCALAR_DIM,
    choose_next_item_largest_first,
    choose_next_item_diverse,
    manager_candidates,
    worker_orientation_candidates,
    worker_scalar_state,
)

_WORKER_CTX: dict[str, Any] | None = None


def _clamp_scale_to_box(
    catalog_dims: list[tuple[float, float, float]],
    catalog_idx: int,
    requested_scale: float,
    box_size: tuple[float, float, float],
) -> float:
    """Reduce scale if the object at requested_scale would exceed any box dimension."""
    base = catalog_dims[catalog_idx]
    max_scale = min(box_size[i] / max(float(base[i]), 1e-8) for i in range(3))
    return float(min(requested_scale, max_scale))


def packed_parts_volume(env: PackingEnv, packed_ids: list[int]) -> float:
    """Return total actual mesh volume of packed parts (not AABB volume)."""
    vol = 0.0
    for item_id in packed_ids:
        vol += env.item_volume(int(item_id))
    return float(vol)


def build_episode_pool(
    catalog_size: int,
    num_objects: int,
    num_combos: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    combos: list[list[int]] = []
    replace = num_objects > catalog_size
    for _ in range(num_combos):
        ids = rng.choice(catalog_size, size=num_objects, replace=replace)
        combos.append([int(x) for x in ids])
    return combos


def select_worker_action_scoremap(
    q_net: WorkerQNet,
    orientation_candidates: list[dict],
    env: PackingEnv,
    rng: np.random.Generator,
    epsilon: float,
    device: torch.device,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray, np.ndarray | torch.Tensor]:
    if len(orientation_candidates) == 0:
        raise ValueError("worker orientation candidates are empty.")

    if rng.random() < epsilon:
        ori_idx = int(rng.integers(0, len(orientation_candidates)))
        cand = orientation_candidates[ori_idx]
        if "legal_mask_t" in cand:
            legal_xy_t = torch.nonzero(cand["legal_mask_t"], as_tuple=False)
            pick_t = legal_xy_t[int(rng.integers(0, int(legal_xy_t.shape[0])))]
            z_t = cand["z_map_t"][pick_t[0], pick_t[1]]
            xyz = torch.stack(
                [
                    pick_t[0].to(torch.float32),
                    pick_t[1].to(torch.float32),
                    z_t.to(torch.float32),
                ],
                dim=0,
            )
            action_t = torch.cat([cand["ori_t"], xyz], dim=0)
            xyz_cpu = xyz.detach().cpu().numpy()
            x, y = int(xyz_cpu[0]), int(xyz_cpu[1])
            z = float(xyz_cpu[2])
        else:
            legal_xy = np.argwhere(cand["legal_mask"] > 0.5)
            pick = legal_xy[int(rng.integers(0, len(legal_xy)))]
            x, y = int(pick[0]), int(pick[1])
            z = float(cand["z_map"][x, y])
    else:
        with torch.no_grad():
            if "map_state_t" in orientation_candidates[0]:
                batch_cache = orientation_candidates[0].get("_batch_cache")
                if batch_cache is not None:
                    maps = batch_cache["maps_t"]
                    legal_masks = batch_cache["legal_masks_t"]
                    z_maps = batch_cache["z_maps_t"]
                    ori_batch_t = batch_cache["ori_t"]
                else:
                    maps = torch.stack([c["map_state_t"] for c in orientation_candidates], dim=0)
                    legal_masks = torch.stack([c["legal_mask_t"] for c in orientation_candidates], dim=0)
                    z_maps = torch.stack([c["z_map_t"] for c in orientation_candidates], dim=0)
                    ori_batch_t = torch.stack([c["ori_t"] for c in orientation_candidates], dim=0)
            else:
                maps = torch.from_numpy(np.stack([c["map_state"] for c in orientation_candidates], axis=0)).to(device)
                legal_masks = torch.from_numpy(
                    np.stack([c["legal_mask"] > 0.5 for c in orientation_candidates], axis=0)
                ).to(device)
                z_maps = None
                ori_batch_t = None
            scores = q_net.score_map(maps).squeeze(1)
            neg_inf = torch.finfo(scores.dtype).min
            masked = scores.masked_fill(~legal_masks, neg_inf)
            flat_t = torch.argmax(masked)
            wdim = int(masked.shape[2])
            hw = int(masked.shape[1] * masked.shape[2])
            ori_t = torch.div(flat_t, hw, rounding_mode="floor")
            rem_t = flat_t - ori_t * hw
            x_t = torch.div(rem_t, wdim, rounding_mode="floor")
            y_t = rem_t - x_t * wdim
            if z_maps is not None:
                z_t = z_maps[ori_t, x_t, y_t]
                vals_t = torch.stack(
                    [
                        ori_t.to(torch.float32),
                        x_t.to(torch.float32),
                        y_t.to(torch.float32),
                        z_t.to(torch.float32),
                    ],
                    dim=0,
                )
                assert ori_batch_t is not None
                action_t = torch.cat([ori_batch_t[ori_t], vals_t[1:4]], dim=0)
                vals = vals_t.detach().cpu().numpy()
                ori_idx = int(vals[0])
                x = int(vals[1])
                y = int(vals[2])
                z = float(vals[3])
            else:
                flat = int(flat_t.item())
                ori_idx = int(flat // hw)
                rem = int(flat % hw)
                x = int(rem // wdim)
                y = int(rem % wdim)
                z = float(orientation_candidates[ori_idx]["z_map"][x, y])
                action_t = None

    chosen = orientation_candidates[ori_idx]
    ori = chosen["ori"]
    scalar = worker_scalar_state(env, x=x, y=y, z=z, w=chosen["w"], h=chosen["h"], ori=ori)
    if "ori_t" in chosen:
        if "action_t" not in locals() or action_t is None:
            action_t = torch.tensor([ori[0], ori[1], ori[2], float(x), float(y), float(z)], dtype=torch.float32, device=device)
        action = action_t
    else:
        action = np.array([ori[0], ori[1], ori[2], x, y, z], dtype=np.float64)
    if "map_state_t" in chosen:
        chosen_map = chosen["map_state_t"].detach()
    else:
        chosen_map = chosen["map_state"]
    return chosen_map, scalar, action


def _clone_state_array(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return x.copy()


def max_worker_q_scoremap(
    q_net: WorkerQNet,
    orientation_candidates: list[dict],
    device: torch.device,
) -> float:
    if len(orientation_candidates) == 0:
        return 0.0
    with torch.no_grad():
        if "map_state_t" in orientation_candidates[0]:
            batch_cache = orientation_candidates[0].get("_batch_cache")
            if batch_cache is not None:
                maps = batch_cache["maps_t"]
                legal_masks = batch_cache["legal_masks_t"]
            else:
                maps = torch.stack([c["map_state_t"] for c in orientation_candidates], dim=0)
                legal_masks = torch.stack([c["legal_mask_t"] for c in orientation_candidates], dim=0)
        else:
            maps = torch.from_numpy(np.stack([c["map_state"] for c in orientation_candidates], axis=0)).to(device)
            legal_masks = torch.from_numpy(
                np.stack([c["legal_mask"] > 0.5 for c in orientation_candidates], axis=0)
            ).to(device)
        scores = q_net.score_map(maps).squeeze(1)
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~legal_masks, neg_inf)
        best = float(torch.max(masked).item())
    if best <= (-1e30):
        return 0.0
    return best


def _init_rollout_worker(init_cfg: dict[str, Any]) -> None:
    global _WORKER_CTX

    seed = int(init_cfg["seed"])
    set_seed(seed)
    device = torch.device("cpu")

    _box_size_cfg = init_cfg.get("box_size", [0.256, 0.256, 0.256])
    env = PackingEnv(
        obj_dir=init_cfg["obj_dir"],
        is_gui=False,
        box_size=(float(_box_size_cfg[0]), float(_box_size_cfg[1]), float(_box_size_cfg[2])),
        resolution=int(init_cfg["resolution"]),
        seed=seed,
        gravity=(0.0, 0.0, float(init_cfg["gravity_z"])),
        use_cuda_state=False,
    )
    manager_q = ManagerQNet(
        hidden_dim=int(init_cfg["manager_hidden_dim"]),
        map_channels=MANAGER_MAP_CHANNELS,
        scalar_dim=MANAGER_SCALAR_DIM,
    ).to(device)
    manager_t = ManagerQNet(
        hidden_dim=int(init_cfg["manager_hidden_dim"]),
        map_channels=MANAGER_MAP_CHANNELS,
        scalar_dim=MANAGER_SCALAR_DIM,
    ).to(device)
    worker_q = WorkerQNet(
        hidden_dim=int(init_cfg["worker_hidden_dim"]),
        map_channels=WORKER_MAP_CHANNELS,
        scalar_dim=WORKER_SCALAR_DIM,
    ).to(device)
    worker_t = WorkerQNet(
        hidden_dim=int(init_cfg["worker_hidden_dim"]),
        map_channels=WORKER_MAP_CHANNELS,
        scalar_dim=WORKER_SCALAR_DIM,
    ).to(device)
    manager_q.eval()
    manager_t.eval()
    worker_q.eval()
    worker_t.eval()

    reward_weights = RewardWeights(
        compactness=float(init_cfg["w_compactness"]),
        pyramidality=float(init_cfg["w_pyramidality"]),
        delta_compactness_bonus=float(init_cfg["w_delta_compactness"]),
        delta_pyramidality_bonus=float(init_cfg["w_delta_pyramidality"]),
        step_density_bonus=float(init_cfg["w_step_density"]),
        height_growth_penalty=float(init_cfg["w_height_penalty"]),
    )

    _WORKER_CTX = {
        "device": device,
        "env": env,
        "manager_q": manager_q,
        "manager_t": manager_t,
        "worker_q": worker_q,
        "worker_t": worker_t,
        "reward_weights": reward_weights,
    }


def rollout_episode(
    env: PackingEnv,
    manager_q,
    manager_t,
    worker_q,
    worker_t,
    rng: np.random.Generator,
    episode_ids: list[int],
    episode_scales: list[float],
    epsilon: float,
    is_joint_phase: bool,
    max_steps: int,
    grid_step: int,
    orientation_step: float,
    manager_top_k: int,
    reward_weights: RewardWeights,
    device: torch.device,
    use_diverse_pretrain: bool = True,
) -> dict[str, Any]:
    ep_start = time.time()
    state = env.reset(object_ids=episode_ids, object_scales=episode_scales)
    ep_reward = 0.0
    ep_compact_gains: list[float] = []
    ep_pyramid_gains: list[float] = []
    ep_height_growths: list[float] = []
    manager_transitions: list[QTransition] = []
    worker_transitions: list[QTransition] = []

    # Cache for worker candidates to avoid recomputation
    # Format: (item_id, candidates) - reused if next iteration selects same item
    cached_w_cands: tuple[int, list[dict]] | None = None

    done = False
    for _ in range(max_steps):
        unpacked = state["unpacked"]
        if len(unpacked) == 0:
            done = True
            break

        if is_joint_phase:
            m_cands = manager_candidates(env, unpacked, max_objects_k=manager_top_k)
            m_maps = np.stack([c[0] for c in m_cands], axis=0).astype(np.float32)
            m_scalars = np.stack([c[1] for c in m_cands], axis=0).astype(np.float32)
            m_items = [c[2] for c in m_cands]

            if rng.random() < epsilon:
                m_idx = int(rng.integers(0, len(m_items)))
            else:
                m_idx = argmax_q_index(manager_q, m_maps, m_scalars, device)
            chosen_m_map = m_maps[m_idx]
            chosen_m_scalar = m_scalars[m_idx]
            item_id = m_items[m_idx]
        else:
            chosen_m_map = None
            chosen_m_scalar = None
            # Use diverse orderings during pre-training for better worker generalization
            if use_diverse_pretrain:
                item_id = choose_next_item_diverse(env, unpacked, rng)
            else:
                item_id = choose_next_item_largest_first(env, unpacked)

        # Use cached worker candidates if available and matches selected item
        if cached_w_cands is not None and cached_w_cands[0] == item_id:
            w_cands = cached_w_cands[1]
            cached_w_cands = None  # Consume the cache
        else:
            w_cands = worker_orientation_candidates(
                env,
                item_id,
                grid_step=grid_step,
                orientation_step=orientation_step,
                use_cuda=(device.type == "cuda"),
            )
        if len(w_cands) == 0:
            break
        chosen_w_map, chosen_w_scalar, action = select_worker_action_scoremap(
            q_net=worker_q,
            orientation_candidates=w_cands,
            env=env,
            rng=rng,
            epsilon=epsilon,
            device=device,
        )

        prev_hm = state["box_heightmap"]
        prev_packed_vol = packed_parts_volume(env, state["packed"])
        next_state, _raw_reward, done, info = env.step(item_id, action)
        next_packed_vol = packed_parts_volume(env, next_state["packed"])
        reward, reward_info = transition_reward(
            prev_heightmap=prev_hm,
            next_heightmap=next_state["box_heightmap"],
            prev_packed_parts_volume=prev_packed_vol,
            next_packed_parts_volume=next_packed_vol,
            stable=bool(info.get("stable", False)),
            valid_action=bool(info.get("valid_action", True)),
            box_size=env.box_size,
            weights=reward_weights,
        )
        ep_reward += reward
        ep_compact_gains.append(float(reward_info.get("compact_gain", 0.0)))
        ep_pyramid_gains.append(float(reward_info.get("pyramid_gain", 0.0)))
        ep_height_growths.append(float(reward_info.get("height_growth", 0.0)))

        if done or len(next_state["unpacked"]) == 0:
            next_m_max_q = 0.0
            next_w_max_q = 0.0
            cached_w_cands = None  # Clear cache on episode end
        else:
            if is_joint_phase:
                nm_cands = manager_candidates(env, next_state["unpacked"], max_objects_k=manager_top_k)
                nm_maps = np.stack([c[0] for c in nm_cands], axis=0).astype(np.float32)
                nm_scalars = np.stack([c[1] for c in nm_cands], axis=0).astype(np.float32)
                next_m_max_q = max_q(manager_t, nm_maps, nm_scalars, device)
                nm_idx = argmax_q_index(manager_t, nm_maps, nm_scalars, device)
                next_item = nm_cands[nm_idx][2]
            else:
                next_m_max_q = 0.0
                next_item = choose_next_item_largest_first(env, next_state["unpacked"])
            nw_cands = worker_orientation_candidates(
                env,
                next_item,
                grid_step=grid_step,
                orientation_step=orientation_step,
                use_cuda=(device.type == "cuda"),
            )
            if len(nw_cands) == 0:
                next_w_max_q = 0.0
                cached_w_cands = None
            else:
                next_w_max_q = max_worker_q_scoremap(worker_t, nw_cands, device)
                # Cache for potential reuse in next iteration
                cached_w_cands = (next_item, nw_cands)

        if is_joint_phase and chosen_m_map is not None and chosen_m_scalar is not None:
            manager_transitions.append(
                QTransition(
                    map_state=_clone_state_array(chosen_m_map),
                    scalar_state=chosen_m_scalar.copy(),
                    reward=float(reward),
                    next_max_q=float(next_m_max_q),
                    done=bool(done),
                )
            )
        worker_transitions.append(
            QTransition(
                map_state=_clone_state_array(chosen_w_map),
                scalar_state=chosen_w_scalar.copy(),
                reward=float(reward),
                next_max_q=float(next_w_max_q),
                done=bool(done),
            )
        )

        state = next_state
        if done:
            break

    success = 1.0 if len(state["unpacked"]) == 0 else 0.0
    return {
        "reward": float(ep_reward),
        "success": float(success),
        "mean_compact_gain": float(np.mean(ep_compact_gains)) if ep_compact_gains else 0.0,
        "mean_pyramid_gain": float(np.mean(ep_pyramid_gains)) if ep_pyramid_gains else 0.0,
        "mean_height_growth": float(np.mean(ep_height_growths)) if ep_height_growths else 0.0,
        "manager_transitions": manager_transitions,
        "worker_transitions": worker_transitions,
        "elapsed_sec": float(time.time() - ep_start),
    }


def _to_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _serialize_transition(t: QTransition) -> dict[str, Any]:
    return {
        "map_state": t.map_state.detach().cpu().numpy() if isinstance(t.map_state, torch.Tensor) else t.map_state,
        "scalar_state": t.scalar_state,
        "reward": float(t.reward),
        "next_max_q": float(t.next_max_q),
        "done": bool(t.done),
    }


def _deserialize_transition(payload: dict[str, Any]) -> QTransition:
    if isinstance(payload, QTransition):
        return payload
    return QTransition(
        map_state=np.asarray(payload["map_state"]),
        scalar_state=np.asarray(payload["scalar_state"]),
        reward=float(payload["reward"]),
        next_max_q=float(payload["next_max_q"]),
        done=bool(payload["done"]),
    )


def rollout_worker_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CTX is None:
        raise RuntimeError("Worker context not initialized. Expected initializer to run.")

    seed = int(task["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    manager_q = _WORKER_CTX["manager_q"]
    manager_t = _WORKER_CTX["manager_t"]
    worker_q = _WORKER_CTX["worker_q"]
    worker_t = _WORKER_CTX["worker_t"]

    manager_q.load_state_dict(task["manager_q_state_dict"])
    manager_t.load_state_dict(task["manager_t_state_dict"])
    worker_q.load_state_dict(task["worker_q_state_dict"])
    worker_t.load_state_dict(task["worker_t_state_dict"])

    result = rollout_episode(
        env=_WORKER_CTX["env"],
        manager_q=manager_q,
        manager_t=manager_t,
        worker_q=worker_q,
        worker_t=worker_t,
        rng=rng,
        episode_ids=task["episode_ids"],
        episode_scales=task["episode_scales"],
        epsilon=float(task["epsilon"]),
        is_joint_phase=bool(task["is_joint_phase"]),
        max_steps=int(task["max_steps"]),
        grid_step=int(task["grid_step"]),
        orientation_step=float(task["orientation_step"]),
        manager_top_k=int(task["manager_top_k"]),
        reward_weights=_WORKER_CTX["reward_weights"],
        device=_WORKER_CTX["device"],
        use_diverse_pretrain=bool(task.get("use_diverse_pretrain", True)),
    )

    return {
        "ep": int(task["ep"]),
        "reward": float(result["reward"]),
        "success": float(result["success"]),
        "mean_compact_gain": float(result["mean_compact_gain"]),
        "mean_pyramid_gain": float(result["mean_pyramid_gain"]),
        "mean_height_growth": float(result["mean_height_growth"]),
        "elapsed_sec": float(result["elapsed_sec"]),
        "manager_transitions": [_serialize_transition(t) for t in result["manager_transitions"]],
        "worker_transitions": [_serialize_transition(t) for t in result["worker_transitions"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical packing policy (manager + worker Q-networks).")
    parser.add_argument("--obj_dir", type=str, default="pybullet-object-models-master")
    parser.add_argument("--num_objects", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--stage1_episodes", type=int, default=100)
    parser.add_argument("--manager_update_interval_epochs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--gravity_z", type=float, default=0.0)
    parser.add_argument("--grid_step", type=int, default=4)
    parser.add_argument("--orientation_step", type=float, default=PAPER_ORIENTATION_STEP)
    parser.add_argument("--max_candidates", type=int, default=512)
    parser.add_argument("--manager_top_k", type=int, default=PAPER_MANAGER_TOP_K)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_size", type=int, default=30000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--manager_lr", type=float, default=1e-3)
    parser.add_argument("--worker_lr", type=float, default=1e-3)
    parser.add_argument("--manager_hidden_dim", type=int, default=128)
    parser.add_argument("--worker_hidden_dim", type=int, default=256)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--eps_start", type=float, default=0.9)
    parser.add_argument("--eps_end", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=float, default=60.0,
                        help="Epsilon decay rate (lower = faster decay)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--episode_pool_size", type=int, default=5000)
    parser.add_argument("--scale_min", type=float, default=0.8)
    parser.add_argument("--scale_max", type=float, default=1.2)
    parser.add_argument("--box_size", type=float, nargs=3, default=[0.256, 0.256, 0.256],
                        metavar=("W", "D", "H"),
                        help="Packing box dimensions in metres (width depth height). Default: 0.256 0.256 0.256")
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--allow_secondary_multiprocess",
        action="store_true",
        default=False,
        help="Allow secondary multiprocessing rollout path (primarily CPU workers).",
    )
    parser.add_argument("--save_path", type=str, default="logs/packing_hrl.pt")
    parser.add_argument("--log_every", type=int, default=10)
    # SLS-optimised reward weights
    parser.add_argument("--w_compactness", type=float, default=0.5,
                        help="Objective weight for compactness (parts_vol / bbox_vol)")
    parser.add_argument("--w_pyramidality", type=float, default=0.3,
                        help="Objective weight for pyramidality (parts_vol / occupied_vol)")
    parser.add_argument("--w_delta_compactness", type=float, default=0.40,
                        help="Per-step reward weight for compactness improvement")
    parser.add_argument("--w_delta_pyramidality", type=float, default=0.40,
                        help="Per-step reward weight for pyramidality improvement")
    parser.add_argument("--w_step_density", type=float, default=0.20,
                        help="Per-step reward weight for current packing density (non-telescoping)")
    parser.add_argument("--w_height_penalty", type=float, default=0.10,
                        help="Per-step penalty weight for height growth (lower build = less SLS print time)")
    # New training efficiency arguments
    parser.add_argument("--updates_per_episode", type=int, default=8,
                        help="Fixed number of gradient updates per episode (decoupled from transitions)")
    parser.add_argument("--target_update_episodes", type=int, default=5,
                        help="Update target networks every N episodes")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine", "step"],
                        help="Learning rate schedule type")
    parser.add_argument("--lr_warmup_episodes", type=int, default=20,
                        help="Number of warmup episodes before LR scheduling")
    parser.add_argument("--diverse_pretrain", action="store_true", default=True,
                        help="Use diverse item orderings during pre-training phase")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save a resumable checkpoint every N episodes (0 to disable)")
    parser.add_argument("--checkpoint_path", type=str, default="logs/checkpoint_hrl.pt",
                        help="Path for periodic resumable checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")
    parser.add_argument("--checkpoint_with_replay", action="store_true", default=False,
                        help="Include replay buffers in checkpoints (WARNING: can be many GB at large replay sizes; "
                             "omit to save only model/optimizer state and refill the buffer after resume)")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.num_workers > 1 and args.gui:
        raise ValueError("--gui is not supported with --num_workers > 1.")

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    print(f"training_start={time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    print(f"device={device}")

    if device.type == "cuda" and args.num_workers > 1 and not args.allow_secondary_multiprocess:
        print("note=single_process_cuda_primary forcing_num_workers=1 (set --allow_secondary_multiprocess to override)")
        args.num_workers = 1
    elif device.type == "cuda" and args.num_workers > 1:
        print("note=secondary_multiprocess_path enabled; rollout workers run on CPU and add transfer/serialization overhead")

    env = PackingEnv(
        obj_dir=args.obj_dir,
        is_gui=args.gui,
        box_size=tuple(args.box_size),
        resolution=args.resolution,
        seed=args.seed,
        gravity=(0.0, 0.0, args.gravity_z),
        use_cuda_state=(device.type == "cuda"),
    )
    # Cache catalog dims and box size for scale clamping in both single/multi worker paths
    catalog_dims: list[tuple[float, float, float]] = list(env._catalog_dims)
    box_size: tuple[float, float, float] = env.box_size

    manager_map_channels = MANAGER_MAP_CHANNELS
    manager_scalar_dim = MANAGER_SCALAR_DIM
    worker_map_channels = WORKER_MAP_CHANNELS
    worker_scalar_dim = WORKER_SCALAR_DIM

    manager_q = ManagerQNet(
        hidden_dim=args.manager_hidden_dim,
        map_channels=manager_map_channels,
        scalar_dim=manager_scalar_dim,
    ).to(device)
    manager_t = ManagerQNet(
        hidden_dim=args.manager_hidden_dim,
        map_channels=manager_map_channels,
        scalar_dim=manager_scalar_dim,
    ).to(device)
    manager_t.load_state_dict(manager_q.state_dict())
    manager_t.eval()

    worker_q = WorkerQNet(
        hidden_dim=args.worker_hidden_dim,
        map_channels=worker_map_channels,
        scalar_dim=worker_scalar_dim,
    ).to(device)
    worker_t = WorkerQNet(
        hidden_dim=args.worker_hidden_dim,
        map_channels=worker_map_channels,
        scalar_dim=worker_scalar_dim,
    ).to(device)
    worker_t.load_state_dict(worker_q.state_dict())
    worker_t.eval()

    manager_opt = torch.optim.Adam(manager_q.parameters(), lr=args.manager_lr)
    worker_opt = torch.optim.Adam(worker_q.parameters(), lr=args.worker_lr)

    # Learning rate schedulers
    if args.lr_schedule == "cosine":
        # Cosine annealing from initial LR to near-zero over training
        manager_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            manager_opt, T_max=max(1, args.episodes - args.lr_warmup_episodes), eta_min=1e-6
        )
        worker_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            worker_opt, T_max=max(1, args.episodes - args.lr_warmup_episodes), eta_min=1e-6
        )
    elif args.lr_schedule == "step":
        # Step decay: reduce by 0.5 every 1/3 of episodes
        step_size = max(1, args.episodes // 3)
        manager_scheduler = torch.optim.lr_scheduler.StepLR(manager_opt, step_size=step_size, gamma=0.5)
        worker_scheduler = torch.optim.lr_scheduler.StepLR(worker_opt, step_size=step_size, gamma=0.5)
    else:
        manager_scheduler = None
        worker_scheduler = None

    replay = HierarchicalReplay(
        manager=ReplayBuffer(
            args.replay_size,
            storage_device=(device if device.type == "cuda" else None),
            allow_cpu_fallback_on_oom=True,
        ),
        worker=ReplayBuffer(
            args.replay_size,
            storage_device=(device if device.type == "cuda" else None),
            allow_cpu_fallback_on_oom=True,
        ),
    )
    manager_replay = replay.manager
    worker_replay = replay.worker

    reward_weights = RewardWeights(
        compactness=args.w_compactness,
        pyramidality=args.w_pyramidality,
        delta_compactness_bonus=args.w_delta_compactness,
        delta_pyramidality_bonus=args.w_delta_pyramidality,
        step_density_bonus=args.w_step_density,
        height_growth_penalty=args.w_height_penalty,
    )

    rng = np.random.default_rng(args.seed)
    episode_pool = build_episode_pool(
        catalog_size=env.catalog_size,
        num_objects=args.num_objects,
        num_combos=max(1, args.episode_pool_size),
        seed=args.seed,
    )
    global_step = 0
    best_success = -1.0
    recent_success: deque[float] = deque(maxlen=50)
    stage1_episodes = int(max(0, min(args.stage1_episodes, args.episodes)))
    manager_update_interval = int(max(1, args.manager_update_interval_epochs))
    start_ep = 0

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if args.resume is not None:
        print(f"resume=loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        manager_q.load_state_dict(ckpt["manager_q_state_dict"])
        manager_t.load_state_dict(ckpt["manager_t_state_dict"])
        worker_q.load_state_dict(ckpt["worker_q_state_dict"])
        worker_t.load_state_dict(ckpt["worker_t_state_dict"])
        manager_opt.load_state_dict(ckpt["manager_opt_state_dict"])
        worker_opt.load_state_dict(ckpt["worker_opt_state_dict"])
        if manager_scheduler is not None and "manager_scheduler_state_dict" in ckpt:
            manager_scheduler.load_state_dict(ckpt["manager_scheduler_state_dict"])
        if worker_scheduler is not None and "worker_scheduler_state_dict" in ckpt:
            worker_scheduler.load_state_dict(ckpt["worker_scheduler_state_dict"])
        if "manager_replay_state_dict" in ckpt:
            manager_replay.load_state_dict(ckpt["manager_replay_state_dict"])
        if "worker_replay_state_dict" in ckpt:
            worker_replay.load_state_dict(ckpt["worker_replay_state_dict"])
        if "rng_state" in ckpt:
            rng = np.random.default_rng()
            rng.bit_generator.state = ckpt["rng_state"]
        start_ep = int(ckpt.get("next_ep", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_success = float(ckpt.get("best_success", -1.0))
        for v in ckpt.get("recent_success", []):
            recent_success.append(float(v))
        print(f"resume=ok next_ep={start_ep} global_step={global_step} best_success={best_success:.4f}")

    def save_checkpoint(ep_idx: int, path: str) -> None:
        """Save a full resumable checkpoint (all state needed to continue training)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        ckpt_data: dict = {
                "next_ep": ep_idx + 1,
                "global_step": global_step,
                "best_success": best_success,
                "recent_success": list(recent_success),
                "manager_q_state_dict": manager_q.state_dict(),
                "manager_t_state_dict": manager_t.state_dict(),
                "worker_q_state_dict": worker_q.state_dict(),
                "worker_t_state_dict": worker_t.state_dict(),
                "manager_opt_state_dict": manager_opt.state_dict(),
                "worker_opt_state_dict": worker_opt.state_dict(),
                "manager_scheduler_state_dict": manager_scheduler.state_dict() if manager_scheduler is not None else None,
                "worker_scheduler_state_dict": worker_scheduler.state_dict() if worker_scheduler is not None else None,
                "rng_state": rng.bit_generator.state,
                "args": vars(args),
        }
        if args.checkpoint_with_replay:
            ckpt_data["manager_replay_state_dict"] = manager_replay.state_dict()
            ckpt_data["worker_replay_state_dict"] = worker_replay.state_dict()
        torch.save(ckpt_data, tmp_path)
        # Atomic replace: avoids a corrupt checkpoint if interrupted mid-write
        os.replace(tmp_path, path)
        print(f"checkpoint_saved ep={ep_idx+1} path={path}")

    def process_episode_result(ep_idx: int, ep_result: dict[str, Any]) -> None:
        nonlocal best_success, global_step

        manager_transitions = [
            _deserialize_transition(t) for t in ep_result["manager_transitions"]
        ]
        worker_transitions = [
            _deserialize_transition(t) for t in ep_result["worker_transitions"]
        ]

        for t in manager_transitions:
            manager_replay.add(t)
        for t in worker_transitions:
            worker_replay.add(t)

        is_joint_phase = ep_idx >= stage1_episodes
        joint_epoch_idx = max(0, ep_idx - stage1_episodes)
        manager_update_due = is_joint_phase and (joint_epoch_idx % manager_update_interval == 0)

        m_losses: list[float] = []
        w_losses: list[float] = []

        # Fixed number of gradient updates per episode (decoupled from transition count)
        num_updates = args.updates_per_episode

        # Manager updates (only in joint phase, on scheduled intervals)
        if manager_update_due:
            for _ in range(num_updates):
                ml = optimize_step(
                    q_net=manager_q,
                    optimizer=manager_opt,
                    replay=manager_replay,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    device=device,
                    max_grad_norm=args.max_grad_norm,
                )
                if ml is not None:
                    m_losses.append(ml)

        # Worker updates (always, fixed count per episode)
        for _ in range(num_updates):
            wl = optimize_step(
                q_net=worker_q,
                optimizer=worker_opt,
                replay=worker_replay,
                batch_size=args.batch_size,
                gamma=args.gamma,
                device=device,
                max_grad_norm=args.max_grad_norm,
            )
            if wl is not None:
                w_losses.append(wl)
            global_step += 1

        # Target network update (episodic, not step-based)
        if (ep_idx + 1) % args.target_update_episodes == 0:
            manager_t.load_state_dict(manager_q.state_dict())
            worker_t.load_state_dict(worker_q.state_dict())

        # LR scheduler step (after warmup, only when the optimizer was actually used
        # this episode — avoids PyTorch's "scheduler.step() before optimizer.step()"
        # warning that fires when the replay buffer is too small to produce a batch)
        if ep_idx >= args.lr_warmup_episodes:
            if worker_scheduler is not None and w_losses:
                worker_scheduler.step()
            if manager_scheduler is not None and is_joint_phase and m_losses:
                manager_scheduler.step()

        # Periodic checkpoint (saved before the best-model check so we always have
        # a recent resumable state even if moving_success never improves)
        if args.checkpoint_every > 0 and (ep_idx + 1) % args.checkpoint_every == 0:
            save_checkpoint(ep_idx, args.checkpoint_path)

        success = float(ep_result["success"])
        recent_success.append(success)
        moving_success = float(np.mean(recent_success)) if recent_success else 0.0

        if moving_success >= best_success:
            best_success = moving_success
            torch.save(
                {
                    "manager_state_dict": manager_q.state_dict(),
                    "worker_state_dict": worker_q.state_dict(),
                    "manager_arch": "hybrid",
                    "worker_arch": "hybrid",
                    "manager_map_channels": manager_map_channels,
                    "manager_scalar_dim": manager_scalar_dim,
                    "worker_map_channels": worker_map_channels,
                    "worker_scalar_dim": worker_scalar_dim,
                    "manager_hidden_dim": args.manager_hidden_dim,
                    "worker_hidden_dim": args.worker_hidden_dim,
                    "stage1_episodes": stage1_episodes,
                    "manager_update_interval_epochs": manager_update_interval,
                    "args": vars(args),
                    "global_step": global_step,
                    "moving_success": moving_success,
                },
                args.save_path,
            )

        if (ep_idx + 1) % args.log_every == 0 or ep_idx == 0:
            avg_ml = float(np.mean(m_losses)) if m_losses else float("nan")
            avg_wl = float(np.mean(w_losses)) if w_losses else float("nan")
            print(
                f"ep={ep_idx+1}/{args.episodes} "
                f"phase={'joint' if is_joint_phase else 'worker_pretrain'} "
                f"reward={ep_result['reward']:.3f} "
                f"success={int(success)} "
                f"eps={epsilon_by_episode(ep_idx, args.eps_start, args.eps_end, args.eps_decay):.3f} "
                f"m_loss={avg_ml:.4f} "
                f"w_loss={avg_wl:.4f} "
                f"moving_success={moving_success:.3f} "
                f"m_replay={len(manager_replay)} "
                f"w_replay={len(worker_replay)} "
                f"mean_compact_gain={ep_result.get('mean_compact_gain', 0.0):.5f} "
                f"mean_pyramid_gain={ep_result.get('mean_pyramid_gain', 0.0):.5f} "
                f"mean_height_growth={ep_result.get('mean_height_growth', 0.0):.5f} "
                f"elapsed_sec={ep_result['elapsed_sec']:.2f}"
            )

    if args.num_workers == 1:
        for ep in range(start_ep, args.episodes):
            episode_ids = episode_pool[ep % len(episode_pool)]
            raw_scales = rng.uniform(args.scale_min, args.scale_max, size=len(episode_ids)).astype(np.float64).tolist()
            episode_scales = [_clamp_scale_to_box(catalog_dims, cat_idx, s, box_size) for cat_idx, s in zip(episode_ids, raw_scales)]
            epsilon = epsilon_by_episode(ep, args.eps_start, args.eps_end, args.eps_decay)
            is_joint_phase = ep >= stage1_episodes

            manager_q.eval()
            manager_t.eval()
            worker_q.eval()
            worker_t.eval()
            ep_result = rollout_episode(
                env=env,
                manager_q=manager_q,
                manager_t=manager_t,
                worker_q=worker_q,
                worker_t=worker_t,
                rng=rng,
                episode_ids=episode_ids,
                episode_scales=episode_scales,
                epsilon=epsilon,
                is_joint_phase=is_joint_phase,
                max_steps=args.max_steps,
                grid_step=args.grid_step,
                orientation_step=args.orientation_step,
                manager_top_k=args.manager_top_k,
                reward_weights=reward_weights,
                device=device,
                use_diverse_pretrain=args.diverse_pretrain,
            )

            manager_q.train()
            worker_q.train()
            process_episode_result(ep, ep_result)
    else:
        env.close()
        ctx = mp.get_context("spawn")
        worker_init_cfg = {
            "seed": int(args.seed),
            "obj_dir": args.obj_dir,
            "box_size": list(box_size),
            "resolution": args.resolution,
            "gravity_z": args.gravity_z,
            "manager_hidden_dim": args.manager_hidden_dim,
            "worker_hidden_dim": args.worker_hidden_dim,
            "w_compactness": args.w_compactness,
            "w_pyramidality": args.w_pyramidality,
            "w_delta_compactness": args.w_delta_compactness,
            "w_delta_pyramidality": args.w_delta_pyramidality,
            "w_step_density": args.w_step_density,
            "w_height_penalty": args.w_height_penalty,
            "catalog_dims": [list(d) for d in catalog_dims],
        }

        _mp_catalog_dims: list[tuple[float, float, float]] = [tuple(d) for d in worker_init_cfg["catalog_dims"]]  # type: ignore[misc]
        _mp_box_size: tuple[float, float, float] = tuple(worker_init_cfg["box_size"])  # type: ignore[misc]

        def make_task(ep_idx: int) -> dict[str, Any]:
            episode_ids = episode_pool[ep_idx % len(episode_pool)]
            raw_scales = rng.uniform(args.scale_min, args.scale_max, size=len(episode_ids)).astype(np.float64).tolist()
            episode_scales = [_clamp_scale_to_box(_mp_catalog_dims, cat_idx, s, _mp_box_size) for cat_idx, s in zip(episode_ids, raw_scales)]
            return {
                "ep": ep_idx,
                "seed": int(args.seed + ep_idx + 1),
                "manager_q_state_dict": _to_cpu_state_dict(manager_q),
                "manager_t_state_dict": _to_cpu_state_dict(manager_t),
                "worker_q_state_dict": _to_cpu_state_dict(worker_q),
                "worker_t_state_dict": _to_cpu_state_dict(worker_t),
                "episode_ids": episode_ids,
                "episode_scales": episode_scales,
                "epsilon": epsilon_by_episode(ep_idx, args.eps_start, args.eps_end, args.eps_decay),
                "is_joint_phase": ep_idx >= stage1_episodes,
                "max_steps": args.max_steps,
                "grid_step": args.grid_step,
                "orientation_step": args.orientation_step,
                "manager_top_k": args.manager_top_k,
                "use_diverse_pretrain": args.diverse_pretrain,
            }

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
            initializer=_init_rollout_worker,
            initargs=(worker_init_cfg,),
        ) as executor:
            manager_q.train()
            worker_q.train()
            next_ep = 0
            in_flight: dict[concurrent.futures.Future, int] = {}

            while next_ep < args.episodes and len(in_flight) < args.num_workers:
                future = executor.submit(rollout_worker_task, make_task(next_ep))
                in_flight[future] = next_ep
                next_ep += 1

            while in_flight:
                done, _pending = concurrent.futures.wait(
                    in_flight.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    ep_idx = in_flight.pop(future)
                    result = future.result()
                    process_episode_result(result["ep"], result)

                    if next_ep < args.episodes:
                        nxt = executor.submit(rollout_worker_task, make_task(next_ep))
                        in_flight[nxt] = next_ep
                        next_ep += 1

    if args.num_workers == 1:
        env.close()
    print(f"training_done save_path={args.save_path} best_moving_success={best_success:.3f}")


if __name__ == "__main__":
    main()
