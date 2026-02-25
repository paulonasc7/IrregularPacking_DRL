#!/usr/bin/env python

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from packing import PackingEnv
from packing.state import (
    MANAGER_MAP_CHANNELS,
    PAPER_MANAGER_TOP_K,
    PAPER_ORIENTATION_STEP,
    WORKER_MAP_CHANNELS,
    choose_next_item_largest_first,
    manager_candidates,
    worker_orientation_candidates,
)


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


def item_aabb_features(env: PackingEnv, item_id: int) -> tuple[float, float, float, float]:
    dims = np.maximum(np.asarray(env.item_dims(item_id), dtype=np.float64), 1e-6)
    volume = float(np.prod(dims))
    return float(dims[0]), float(dims[1]), float(dims[2]), volume


def build_feature_vector(
    env: PackingEnv,
    box_hm: np.ndarray,
    ht: np.ndarray,
    hb: np.ndarray,
    ori: np.ndarray,
    x: int,
    y: int,
    z: float,
) -> np.ndarray:
    w, h = ht.shape
    box_h = float(env.box_size[2])
    res = float(env.resolution)
    patch = box_hm[x : x + w, y : y + h]
    updated = np.maximum((ht > 0) * (ht + z), patch)

    footprint = float(np.mean(ht > 0))
    item_height_sum = float(np.sum(np.maximum(ht, 0)))
    item_height_norm = item_height_sum / float(max(1.0, res * res * box_h))

    mean_box = float(np.mean(box_hm))
    max_box = float(np.max(box_hm))
    mean_patch = float(np.mean(patch))
    max_patch = float(np.max(patch))
    mean_updated = float(np.mean(updated))
    max_updated = float(np.max(updated))

    clearance = float(max(0.0, box_h - max_updated))
    comp_delta = float(mean_updated - mean_patch)

    roll, pitch, yaw = [float(v) for v in ori]
    feat = np.array(
        [
            x / res,
            y / res,
            z / max(1e-6, box_h),
            w / res,
            h / res,
            footprint,
            item_height_norm,
            mean_box / max(1e-6, box_h),
            max_box / max(1e-6, box_h),
            mean_patch / max(1e-6, box_h),
            max_patch / max(1e-6, box_h),
            mean_updated / max(1e-6, box_h),
            max_updated / max(1e-6, box_h),
            clearance / max(1e-6, box_h),
            comp_delta / max(1e-6, box_h),
            np.sin(roll),
            np.cos(roll),
            np.sin(pitch),
            np.cos(pitch),
            np.sin(yaw),
            np.cos(yaw),
        ],
        dtype=np.float32,
    )
    return feat


def _normalize_hm(hm: np.ndarray, box_h: float) -> np.ndarray:
    return np.clip(hm / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)


def _center_overlay(canvas: np.ndarray, patch: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    if patch.size == 0:
        return out
    h, w = out.shape
    ph, pw = patch.shape
    if ph <= 0 or pw <= 0 or ph > h or pw > w:
        return out
    x0 = (h - ph) // 2
    y0 = (w - pw) // 2
    out[x0 : x0 + ph, y0 : y0 + pw] = patch
    return out


def generate_candidates(
    env: PackingEnv,
    item_id: int,
    grid_step: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    orientations = env.candidate_orientations(item_id)
    box_hm = env.box_heightmap(copy=False)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []

    for ori in orientations:
        ht, hb = env.item_hm(item_id, ori)
        if ht.size == 0 or hb.size == 0:
            continue
        w, h = ht.shape
        if w > env.resolution or h > env.resolution:
            continue

        x_range = range(0, env.resolution - w + 1, grid_step)
        y_range = range(0, env.resolution - h + 1, grid_step)
        for x in x_range:
            for y in y_range:
                z = float(np.max(box_hm[x : x + w, y : y + h] - hb))
                updated = np.maximum((ht > 0) * (ht + z), box_hm[x : x + w, y : y + h])
                if float(np.max(updated)) > env.box_size[2]:
                    continue
                feat = build_feature_vector(env, box_hm, ht, hb, ori, x, y, z)
                action = np.array([ori[0], ori[1], ori[2], x, y, z], dtype=np.float64)
                candidates.append((feat, action))

    if max_candidates > 0 and len(candidates) > max_candidates:
        idx = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    return candidates


def load_models(model_path: str, cpu: bool) -> tuple[Any, Any, Any, str]:
    import torch
    from packing.models_manager import ManagerQNet
    from packing.models_worker import WorkerQNet

    device = torch.device("cpu")
    if torch.cuda.is_available() and not cpu:
        device = torch.device("cuda")

    ckpt = torch.load(model_path, map_location=device)
    manager_model = None
    worker_model = None

    if "manager_state_dict" in ckpt and "worker_state_dict" in ckpt:
        manager_arch = ckpt.get("manager_arch", "mlp")
        worker_arch = ckpt.get("worker_arch", "mlp")
        manager_hidden_dim = int(ckpt.get("manager_hidden_dim", 128))
        worker_hidden_dim = int(ckpt.get("worker_hidden_dim", 256))

        if manager_arch == "hybrid":
            manager_model = ManagerQNet(
                hidden_dim=manager_hidden_dim,
                map_channels=int(ckpt.get("manager_map_channels", MANAGER_MAP_CHANNELS)),
                scalar_dim=int(ckpt.get("manager_scalar_dim", 8)),
            ).to(device)
        else:
            manager_model = ManagerQNet(int(ckpt.get("manager_feat_dim", 8)), hidden_dim=manager_hidden_dim).to(device)
        manager_model.load_state_dict(ckpt["manager_state_dict"])
        manager_model.eval()

        if worker_arch == "hybrid":
            worker_model = WorkerQNet(
                hidden_dim=worker_hidden_dim,
                map_channels=int(ckpt.get("worker_map_channels", WORKER_MAP_CHANNELS)),
                scalar_dim=int(ckpt.get("worker_scalar_dim", 11)),
            ).to(device)
        else:
            worker_model = WorkerQNet(int(ckpt.get("worker_feat_dim", 21)), hidden_dim=worker_hidden_dim).to(device)
        worker_model.load_state_dict(ckpt["worker_state_dict"])
        worker_model.eval()
        return manager_model, worker_model, device, "hierarchical"

    if "model_state_dict" in ckpt:
        worker_arch = ckpt.get("worker_arch", "mlp")
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        if worker_arch == "hybrid":
            worker_model = WorkerQNet(
                hidden_dim=hidden_dim,
                map_channels=int(ckpt.get("worker_map_channels", WORKER_MAP_CHANNELS)),
                scalar_dim=int(ckpt.get("worker_scalar_dim", 11)),
            ).to(device)
        else:
            feat_dim = int(ckpt.get("feat_dim", 21))
            worker_model = WorkerQNet(feat_dim, hidden_dim=hidden_dim).to(device)
        worker_model.load_state_dict(ckpt["model_state_dict"])
        worker_model.eval()
        return None, worker_model, device, "worker"

    raise ValueError(
        "Unsupported checkpoint format. Expected worker-only keys "
        "('model_state_dict') or hierarchical keys ('manager_state_dict', 'worker_state_dict')."
    )


def choose_item(
    env: PackingEnv,
    unpacked: list[int],
    manager_top_k: int,
    manager_model: Any = None,
    device: Any = None,
) -> tuple[int, str]:
    if manager_model is None:
        return choose_next_item_largest_first(env, unpacked), "heuristic"

    import torch
    if getattr(manager_model, "use_map_encoder", False):
        cands = manager_candidates(env, unpacked, max_objects_k=manager_top_k)
        maps = np.stack([c[0] for c in cands], axis=0).astype(np.float32)
        scalars = np.stack([c[1] for c in cands], axis=0).astype(np.float32)
        items = [c[2] for c in cands]
        map_tensor = torch.from_numpy(maps).to(device)
        scalar_tensor = torch.from_numpy(scalars).to(device)
        with torch.no_grad():
            q_values = manager_model(map_tensor, scalar_tensor).detach().cpu().numpy()
        best_idx = int(np.argmax(q_values))
        return int(items[best_idx]), "model"
    else:
        box_hm = env.box_heightmap(copy=False)
        ucount = len(unpacked)
        feats = []
        for item_id in unpacked:
            dx, dy, dz, volume = item_aabb_features(env, item_id)
            box_h = float(env.box_size[2])
            box_volume = float(env.box_size[0] * env.box_size[1] * env.box_size[2])
            fill_ratio = float(np.sum(box_hm) / max(1e-6, env.resolution * env.resolution * box_h))
            feat = np.array(
                [
                    dx / max(1e-6, env.box_size[0]),
                    dy / max(1e-6, env.box_size[1]),
                    dz / max(1e-6, env.box_size[2]),
                    volume / max(1e-6, box_volume),
                    float(np.mean(box_hm)) / max(1e-6, box_h),
                    float(np.max(box_hm)) / max(1e-6, box_h),
                    fill_ratio,
                    float(ucount) / 64.0,
                ],
                dtype=np.float32,
            )
            feats.append(feat)

        feat_arr = np.stack(feats, axis=0).astype(np.float32)
        with torch.no_grad():
            feat_tensor = torch.from_numpy(feat_arr).to(device)
            q_values = manager_model(feat_tensor).detach().cpu().numpy()
        best_idx = int(np.argmax(q_values))
        return int(unpacked[best_idx]), "model"


def choose_placement(
    env: PackingEnv,
    item_id: int,
    grid_step: int,
    orientation_step: float,
    max_candidates: int,
    rng: np.random.Generator,
    worker_model: Any = None,
    device: Any = None,
) -> tuple[np.ndarray | None, str]:
    if worker_model is None:
        candidates = worker_orientation_candidates(
            env,
            item_id,
            grid_step=grid_step,
            orientation_step=orientation_step,
        )
        if not candidates:
            return None, "none"
        cand = candidates[0]
        legal_xy = np.argwhere(cand["legal_mask"] > 0.5)
        pick = legal_xy[int(rng.integers(0, len(legal_xy)))]
        x, y = int(pick[0]), int(pick[1])
        ori = cand["ori"]
        z = float(cand["z_map"][x, y])
        return np.array([ori[0], ori[1], ori[2], x, y, z], dtype=np.float64), "heuristic"

    import torch
    if getattr(worker_model, "use_map_encoder", False):
        candidates = worker_orientation_candidates(
            env,
            item_id,
            grid_step=grid_step,
            orientation_step=orientation_step,
        )
        if len(candidates) == 0:
            return None, "none"
        maps = np.stack([c["map_state"] for c in candidates], axis=0).astype(np.float32)
        map_tensor = torch.from_numpy(maps).to(device)
        with torch.no_grad():
            score_maps = worker_model.score_map(map_tensor).squeeze(1).detach().cpu().numpy()

        best_val = -1e9
        best_idx = 0
        x = 0
        y = 0
        for i, cand in enumerate(candidates):
            masked = np.where(cand["legal_mask"] > 0.5, score_maps[i], -1e9)
            flat = int(np.argmax(masked))
            score = float(masked.reshape(-1)[flat])
            if score > best_val:
                best_val = score
                best_idx = i
                x = int(flat // masked.shape[1])
                y = int(flat % masked.shape[1])

        chosen = candidates[best_idx]
        ori = chosen["ori"]
        z = float(chosen["z_map"][x, y])
        action = np.array([ori[0], ori[1], ori[2], x, y, z], dtype=np.float64)
        return action, "model"
    else:
        candidates = generate_candidates(env, item_id, grid_step=grid_step, max_candidates=max_candidates, rng=rng)
        if not candidates:
            return None, "none"
        feats = np.stack([c[0] for c in candidates], axis=0).astype(np.float32)
        with torch.no_grad():
            feat_tensor = torch.from_numpy(feats).to(device)
            q_values = worker_model(feat_tensor).detach().cpu().numpy()
        best_idx = int(np.argmax(q_values))
        return candidates[best_idx][1], "model"


def run_episode(
    env: PackingEnv,
    num_objects: int,
    object_ids: list[int] | None,
    object_scales: list[float] | None,
    max_steps: int,
    grid_step: int,
    orientation_step: float,
    max_candidates: int,
    rng: np.random.Generator,
    manager_top_k: int,
    manager_model: Any = None,
    worker_model: Any = None,
    device: Any = None,
) -> dict:
    if object_ids is None:
        state = env.reset(object_ids=None, object_scales=object_scales)
    else:
        episode_object_ids = list(object_ids)
        state = env.reset(object_ids=episode_object_ids, object_scales=object_scales)

    steps = 0
    stable_placements = 0
    manager_model_steps = 0
    placements_from_model = 0
    t0 = time.time()

    while steps < max_steps and len(state["unpacked"]) > 0:
        item_id, manager_source = choose_item(
            env,
            state["unpacked"],
            manager_top_k=manager_top_k,
            manager_model=manager_model,
            device=device,
        )
        if manager_source == "model":
            manager_model_steps += 1
        placement, source = choose_placement(
            env,
            item_id,
            grid_step=grid_step,
            orientation_step=orientation_step,
            max_candidates=max_candidates,
            rng=rng,
            worker_model=worker_model,
            device=device,
        )
        if placement is None:
            break
        if source == "model":
            placements_from_model += 1

        state, reward, done, info = env.step(item_id, placement)
        steps += 1
        if info.get("stable", False):
            stable_placements += 1
        if done:
            break

    elapsed = time.time() - t0
    success = len(state["unpacked"]) == 0

    return {
        "success": success,
        "steps": steps,
        "stable_placements": stable_placements,
        "remaining": len(state["unpacked"]),
        "elapsed_sec": elapsed,
        "manager_model_steps": manager_model_steps,
        "placements_from_model": placements_from_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate packing environment with a simple inference-only baseline.")
    parser.add_argument("--obj_dir", type=str, default="pybullet-object-models-master", help="Object root with objects.csv or URDF tree.")
    parser.add_argument("--num_objects", type=int, default=50, help="Objects per episode.")
    parser.add_argument(
        "--object_ids",
        type=str,
        default="",
        help="Comma-separated catalog indices (e.g. 0,3,7,11). Overrides --num_objects.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    parser.add_argument("--max_steps", type=int, default=20, help="Step cap per episode.")
    parser.add_argument("--resolution", type=int, default=200, help="Box grid resolution.")
    parser.add_argument("--gravity_z", type=float, default=0.0, help="Gravity along Z in m/s^2. Use 0 for gravity-free packing.")
    parser.add_argument("--grid_step", type=int, default=2, help="Coarse search stride for x/y placement.")
    parser.add_argument("--orientation_step", type=float, default=PAPER_ORIENTATION_STEP, help="Orientation discretization step in radians.")
    parser.add_argument("--manager_top_k", type=int, default=PAPER_MANAGER_TOP_K, help="Top-K objects by bbox volume for manager scoring.")
    parser.add_argument("--max_candidates", type=int, default=512, help="Max candidate placements to score.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic object sampling.")
    parser.add_argument("--episode_pool_size", type=int, default=2000, help="Precomputed random episode combinations.")
    parser.add_argument("--scale_min", type=float, default=0.8, help="Minimum random global scale for objects.")
    parser.add_argument("--scale_max", type=float, default=1.2, help="Maximum random global scale for objects.")
    parser.add_argument("--gui", action="store_true", default=False, help="Enable PyBullet GUI.")
    parser.add_argument("--model_path", type=str, default="", help="Optional worker checkpoint from train_packing.py.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU when loading model.")
    args = parser.parse_args()

    object_ids: list[int] | None = None
    if args.object_ids.strip():
        object_ids = [int(tok.strip()) for tok in args.object_ids.split(",") if tok.strip()]
        if len(object_ids) == 0:
            raise ValueError("--object_ids provided but parsed empty list.")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("--object_ids must be unique.")

    env = PackingEnv(
        obj_dir=args.obj_dir,
        is_gui=args.gui,
        resolution=args.resolution,
        seed=args.seed,
        gravity=(0.0, 0.0, args.gravity_z),
    )
    rng = np.random.default_rng(args.seed)
    episode_pool = build_episode_pool(
        catalog_size=env.catalog_size,
        num_objects=args.num_objects,
        num_combos=max(1, args.episode_pool_size),
        seed=args.seed + 1000,
    )

    model = None
    device = None
    manager_model = None
    worker_model = None
    policy_mode = "heuristic"
    if args.model_path:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
        manager_model, worker_model, device, policy_mode = load_models(args.model_path, cpu=args.cpu)
        print(f"policy={policy_mode} checkpoint={args.model_path}")
    else:
        print("policy=heuristic")

    results = []
    for ep in range(args.episodes):
        if object_ids is None:
            ep_object_ids = episode_pool[ep % len(episode_pool)]
        else:
            ep_object_ids = list(object_ids)
        ep_object_scales = rng.uniform(args.scale_min, args.scale_max, size=len(ep_object_ids)).astype(np.float64).tolist()
        ep_result = run_episode(
            env,
            num_objects=args.num_objects,
            object_ids=ep_object_ids,
            object_scales=ep_object_scales,
            max_steps=args.max_steps,
            grid_step=args.grid_step,
            orientation_step=args.orientation_step,
            max_candidates=args.max_candidates,
            rng=rng,
            manager_top_k=args.manager_top_k,
            manager_model=manager_model,
            worker_model=worker_model,
            device=device,
        )
        results.append(ep_result)
        print(
            f"episode={ep+1} success={ep_result['success']} steps={ep_result['steps']} "
            f"stable={ep_result['stable_placements']} remaining={ep_result['remaining']} "
            f"manager_model_steps={ep_result['manager_model_steps']} "
            f"worker_model_steps={ep_result['placements_from_model']} "
            f"elapsed_sec={ep_result['elapsed_sec']:.3f}"
        )

    env.close()

    success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in results]))
    avg_steps = float(np.mean([r["steps"] for r in results])) if results else 0.0
    avg_time = float(np.mean([r["elapsed_sec"] for r in results])) if results else 0.0

    print("--- summary ---")
    print(f"episodes={len(results)}")
    print(f"success_rate={success_rate:.3f}")
    print(f"avg_steps={avg_steps:.3f}")
    print(f"avg_episode_time_sec={avg_time:.3f}")


if __name__ == "__main__":
    main()
