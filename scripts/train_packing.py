#!/usr/bin/env python

import argparse
import os
import sys
import time
from collections import deque

import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from packing import PackingEnv
from packing.agent_hrl import argmax_q_index, epsilon_by_episode, max_q, optimize_step, set_seed
from packing.models_worker import WorkerQNet
from packing.replay import QTransition, ReplayBuffer
from packing.reward import RewardWeights, transition_reward
from packing.state import (
    PAPER_ORIENTATION_STEP,
    WORKER_MAP_CHANNELS,
    WORKER_SCALAR_DIM,
    choose_next_item_largest_first,
    worker_orientation_candidates,
    worker_scalar_state,
)


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
) -> tuple[np.ndarray | torch.Tensor, np.ndarray, np.ndarray | torch.Tensor, float]:
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
    q_val = float(0.0)
    if not q_net.training:
        with torch.no_grad():
            if "map_state_t" in chosen:
                mt = chosen["map_state_t"][None, ...]
            else:
                mt = torch.from_numpy(chosen["map_state"][None, ...]).to(device)
            st = torch.from_numpy(scalar[None, ...]).to(device)
            q_val = float(q_net(mt, st).item())
    if "map_state_t" in chosen:
        chosen_map = chosen["map_state_t"].detach()
    else:
        chosen_map = chosen["map_state"]
    return chosen_map, scalar, action, q_val


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline worker Q-policy for packing placement.")
    parser.add_argument("--obj_dir", type=str, default="pybullet-object-models-master")
    parser.add_argument("--num_objects", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--gravity_z", type=float, default=0.0)
    parser.add_argument("--grid_step", type=int, default=4)
    parser.add_argument("--orientation_step", type=float, default=PAPER_ORIENTATION_STEP)
    parser.add_argument("--max_candidates", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_size", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--eps_start", type=float, default=0.9)
    parser.add_argument("--eps_end", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--episode_pool_size", type=int, default=5000)
    parser.add_argument("--scale_min", type=float, default=0.8)
    parser.add_argument("--scale_max", type=float, default=1.2)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default="logs/packing_worker_q.pt")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--w_compactness", type=float, default=0.5)
    parser.add_argument("--w_pyramidality", type=float, default=0.3)
    parser.add_argument("--w_stability", type=float, default=0.25)
    parser.add_argument("--step_penalty", type=float, default=0.0)
    parser.add_argument("--invalid_penalty", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    print(f"device={device}")

    env = PackingEnv(
        obj_dir=args.obj_dir,
        is_gui=args.gui,
        resolution=args.resolution,
        seed=args.seed,
        gravity=(0.0, 0.0, args.gravity_z),
        use_cuda_state=(device.type == "cuda"),
    )

    worker_map_channels = WORKER_MAP_CHANNELS
    worker_scalar_dim = WORKER_SCALAR_DIM
    q_net = WorkerQNet(
        hidden_dim=args.hidden_dim,
        map_channels=worker_map_channels,
        scalar_dim=worker_scalar_dim,
    ).to(device)
    target_net = WorkerQNet(
        hidden_dim=args.hidden_dim,
        map_channels=worker_map_channels,
        scalar_dim=worker_scalar_dim,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size, storage_device=(device if device.type == "cuda" else None))

    reward_weights = RewardWeights(
        compactness=args.w_compactness,
        pyramidality=args.w_pyramidality,
        stability=args.w_stability,
        step_penalty=args.step_penalty,
        invalid_penalty=args.invalid_penalty,
    )

    rng = np.random.default_rng(args.seed)
    episode_pool = build_episode_pool(
        catalog_size=env.catalog_size,
        num_objects=args.num_objects,
        num_combos=max(1, args.episode_pool_size),
        seed=args.seed,
    )
    global_step = 0
    best_success_rate = -1.0
    recent_success: deque[float] = deque(maxlen=50)

    for ep in range(args.episodes):
        ep_start = time.time()
        episode_ids = episode_pool[ep % len(episode_pool)]
        episode_scales = rng.uniform(args.scale_min, args.scale_max, size=len(episode_ids)).astype(np.float64).tolist()
        state = env.reset(object_ids=episode_ids, object_scales=episode_scales)
        epsilon = epsilon_by_episode(ep, args.eps_start, args.eps_end, args.eps_decay)

        ep_reward = 0.0
        ep_compact_gains: list[float] = []
        ep_pyramid_gains: list[float] = []
        ep_height_growths: list[float] = []
        ep_losses: list[float] = []
        done = False

        for _ in range(args.max_steps):
            if len(state["unpacked"]) == 0:
                done = True
                break

            item_id = choose_next_item_largest_first(env, state["unpacked"])
            orientation_candidates = worker_orientation_candidates(
                env,
                item_id,
                grid_step=args.grid_step,
                orientation_step=args.orientation_step,
                use_cuda=(device.type == "cuda"),
            )
            if len(orientation_candidates) == 0:
                break
            chosen_map, chosen_scalar, chosen_action, _ = select_worker_action_scoremap(
                q_net=q_net,
                orientation_candidates=orientation_candidates,
                env=env,
                rng=rng,
                epsilon=epsilon,
                device=device,
            )

            prev_hm = state["box_heightmap"]
            prev_packed_vol = packed_parts_volume(env, state["packed"])
            next_state, _raw_reward, done, info = env.step(item_id, chosen_action)
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
                next_max_q = 0.0
            else:
                next_item = choose_next_item_largest_first(env, next_state["unpacked"])
                next_orientation_candidates = worker_orientation_candidates(
                    env,
                    next_item,
                    grid_step=args.grid_step,
                    orientation_step=args.orientation_step,
                    use_cuda=(device.type == "cuda"),
                )
                if len(next_orientation_candidates) == 0:
                    next_max_q = 0.0
                else:
                    next_max_q = max_worker_q_scoremap(target_net, next_orientation_candidates, device)

            replay.add(
                QTransition(
                    map_state=_clone_state_array(chosen_map),
                    scalar_state=chosen_scalar.copy(),
                    reward=float(reward),
                    next_max_q=float(next_max_q),
                    done=bool(done),
                )
            )

            loss_val = optimize_step(
                q_net=q_net,
                optimizer=optimizer,
                replay=replay,
                batch_size=args.batch_size,
                gamma=args.gamma,
                device=device,
            )
            if loss_val is not None:
                ep_losses.append(loss_val)

            global_step += 1
            if global_step % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            state = next_state
            if done:
                break

        success = 1.0 if len(state["unpacked"]) == 0 else 0.0
        recent_success.append(success)
        moving_success = float(np.mean(recent_success)) if recent_success else 0.0

        if moving_success >= best_success_rate:
            best_success_rate = moving_success
            torch.save(
                {
                    "model_state_dict": q_net.state_dict(),
                    "worker_arch": "hybrid",
                    "worker_map_channels": worker_map_channels,
                    "worker_scalar_dim": worker_scalar_dim,
                    "hidden_dim": args.hidden_dim,
                    "args": vars(args),
                    "global_step": global_step,
                    "moving_success": moving_success,
                },
                args.save_path,
            )

        if (ep + 1) % args.log_every == 0 or ep == 0:
            avg_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            print(
                f"ep={ep+1}/{args.episodes} "
                f"reward={ep_reward:.3f} "
                f"success={int(success)} "
                f"eps={epsilon:.3f} "
                f"avg_loss={avg_loss:.4f} "
                f"moving_success={moving_success:.3f} "
                f"replay={len(replay)} "
                f"mean_compact_gain={(float(np.mean(ep_compact_gains)) if ep_compact_gains else 0.0):.5f} "
                f"mean_pyramid_gain={(float(np.mean(ep_pyramid_gains)) if ep_pyramid_gains else 0.0):.5f} "
                f"mean_height_growth={(float(np.mean(ep_height_growths)) if ep_height_growths else 0.0):.5f} "
                f"elapsed_sec={time.time()-ep_start:.2f}"
            )

    env.close()
    print(f"training_done save_path={args.save_path} best_moving_success={best_success_rate:.3f}")


if __name__ == "__main__":
    main()
