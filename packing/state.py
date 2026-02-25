from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch

PAPER_ORIENTATION_STEP = float(np.pi / 2.0)
PAPER_MANAGER_TOP_K = 20

MANAGER_MAP_CHANNELS = 7
MANAGER_SCALAR_DIM = 8
WORKER_MAP_CHANNELS = 3
WORKER_SCALAR_DIM = 11


def aabb_volume(env: Any, item_id: int) -> float:
    dx, dy, dz = env.item_dims(item_id)
    return float(dx * dy * dz)


def item_aabb_features(env: Any, item_id: int) -> tuple[float, float, float, float]:
    dims = np.maximum(np.asarray(env.item_dims(item_id), dtype=np.float64), 1e-6)
    volume = float(np.prod(dims))
    return float(dims[0]), float(dims[1]), float(dims[2]), volume


def choose_next_item_largest_first(env: Any, unpacked: list[int]) -> int:
    candidates = [(aabb_volume(env, item_id), item_id) for item_id in unpacked]
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def choose_next_item_smallest_first(env: Any, unpacked: list[int]) -> int:
    """Select smallest item by volume - useful for diverse pre-training."""
    candidates = [(aabb_volume(env, item_id), item_id) for item_id in unpacked]
    candidates.sort(reverse=False)
    return int(candidates[0][1])


def choose_next_item_random(env: Any, unpacked: list[int], rng: np.random.Generator) -> int:
    """Select random item - useful for diverse pre-training."""
    return int(rng.choice(unpacked))


def choose_next_item_flattest_first(env: Any, unpacked: list[int]) -> int:
    """Select item with smallest height/base ratio - good for stable base layers."""
    def flatness_score(item_id: int) -> float:
        dx, dy, dz = env.item_dims(item_id)
        base_area = max(dx * dy, dx * dz, dy * dz)
        min_dim = min(dx, dy, dz)
        return min_dim / max(1e-6, np.sqrt(base_area))  # lower = flatter
    
    candidates = [(flatness_score(item_id), item_id) for item_id in unpacked]
    candidates.sort(reverse=False)
    return int(candidates[0][1])


def choose_next_item_diverse(
    env: Any, 
    unpacked: list[int], 
    rng: np.random.Generator,
    strategy: str | None = None,
) -> int:
    """Select next item using diverse strategies for robust pre-training.
    
    If strategy is None, randomly chooses between available strategies.
    """
    strategies = ["largest", "smallest", "random", "flattest"]
    if strategy is None:
        strategy = rng.choice(strategies)
    
    if strategy == "largest":
        return choose_next_item_largest_first(env, unpacked)
    elif strategy == "smallest":
        return choose_next_item_smallest_first(env, unpacked)
    elif strategy == "random":
        return choose_next_item_random(env, unpacked, rng)
    elif strategy == "flattest":
        return choose_next_item_flattest_first(env, unpacked)
    else:
        return choose_next_item_largest_first(env, unpacked)


def _normalize_hm(hm: np.ndarray, box_h: float) -> np.ndarray:
    return np.clip(hm / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)


def _center_overlay(canvas: np.ndarray, patch: np.ndarray, fill_value_scale: float = 1.0) -> np.ndarray:
    out = canvas.copy()
    if patch.size == 0:
        return out
    h, w = out.shape
    ph, pw = patch.shape
    if ph <= 0 or pw <= 0 or ph > h or pw > w:
        return out
    x0 = (h - ph) // 2
    y0 = (w - pw) // 2
    out[x0 : x0 + ph, y0 : y0 + pw] = patch * fill_value_scale
    return out


@lru_cache(maxsize=8)
def _orientation_grid_cached(step_key: float) -> tuple:
    """Cached orientation grid computation - returns tuple for hashability."""
    step = float(max(1e-6, step_key))
    values = np.arange(0.0, 2.0 * np.pi, step, dtype=np.float64)
    yaw, pitch, roll = np.meshgrid(values, values, values)
    grid = np.stack([roll.reshape(-1), pitch.reshape(-1), yaw.reshape(-1)], axis=1)
    return tuple(map(tuple, grid.tolist()))


def orientation_grid(step: float = PAPER_ORIENTATION_STEP) -> np.ndarray:
    """Get orientation grid with caching for common step values."""
    # Round step to avoid float precision issues in cache key
    step_key = round(float(step), 8)
    cached = _orientation_grid_cached(step_key)
    return np.array(cached, dtype=np.float64)


def _compute_z_and_legal_maps(
    box_hm: np.ndarray | None,
    ht: np.ndarray | None,
    hb: np.ndarray | None,
    box_h: float,
    stride: int,
    use_cuda: bool = False,
    ht_t: torch.Tensor | None = None,
    hb_t: torch.Tensor | None = None,
    box_t: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Vectorized z-map/legal-mask over stride-sampled top-left placements."""
    if use_cuda:
        if hb_t is None:
            if hb is None:
                raise ValueError("hb or hb_t must be provided for CUDA _compute_z_and_legal_maps.")
            hb_t = torch.from_numpy(np.asarray(hb, dtype=np.float32)).to("cuda")
        w = int(hb_t.shape[0])
        h = int(hb_t.shape[1])
        if w <= 0 or h <= 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                torch.zeros((0, 0), dtype=torch.float32, device="cuda"),
                torch.zeros((0, 0), dtype=torch.bool, device="cuda"),
            )
        if box_t is not None:
            H, W = int(box_t.shape[0]), int(box_t.shape[1])
        elif box_hm is not None:
            H, W = int(box_hm.shape[0]), int(box_hm.shape[1])
        else:
            raise ValueError("box_hm or box_t must be provided for CUDA _compute_z_and_legal_maps.")
        xs = np.arange(0, H - w + 1, max(1, int(stride)), dtype=np.int32)
        ys = np.arange(0, W - h + 1, max(1, int(stride)), dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            return (
                xs,
                ys,
                torch.zeros((0, 0), dtype=torch.float32, device="cuda"),
                torch.zeros((0, 0), dtype=torch.bool, device="cuda"),
            )
        if box_t is None:
            if box_hm is None:
                raise ValueError("box_hm required when box_t is not provided.")
            box_t = torch.from_numpy(np.asarray(box_hm, dtype=np.float32)).to("cuda")
        xs_t = torch.from_numpy(np.asarray(xs, dtype=np.int64)).to("cuda")
        ys_t = torch.from_numpy(np.asarray(ys, dtype=np.int64)).to("cuda")

        windows_t = box_t.unfold(0, w, 1).unfold(1, h, 1)
        sampled_t = windows_t.index_select(0, xs_t).index_select(1, ys_t)
        z_sub_t = (sampled_t - hb_t.view(1, 1, w, h)).amax(dim=(2, 3))
        if ht_t is not None:
            ht_peak_t = torch.amax(torch.clamp_min(ht_t, 0.0))
        elif ht is None:
            ht_peak_t = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        else:
            ht_tmp = torch.from_numpy(np.asarray(ht, dtype=np.float32)).to("cuda")
            ht_peak_t = torch.amax(torch.clamp_min(ht_tmp, 0.0))
        legal_sub_t = (z_sub_t + ht_peak_t) <= (float(box_h) + 1e-6)
        return xs, ys, z_sub_t, legal_sub_t

    if box_hm is None:
        raise ValueError("box_hm must be provided for CPU _compute_z_and_legal_maps.")
    if hb is None:
        raise ValueError("hb must be provided for CPU _compute_z_and_legal_maps.")
    w, h = hb.shape
    if w <= 0 or h <= 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
        )
    windows = np.lib.stride_tricks.sliding_window_view(box_hm, (w, h))
    xs = np.arange(0, windows.shape[0], max(1, int(stride)), dtype=np.int32)
    ys = np.arange(0, windows.shape[1], max(1, int(stride)), dtype=np.int32)
    if xs.size == 0 or ys.size == 0:
        return xs, ys, np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)
    sampled = windows[np.ix_(xs, ys)]
    z_sub = np.max(sampled - hb[None, None, :, :], axis=(2, 3)).astype(np.float32)

    if ht is None:
        ht_pos = np.zeros((0,), dtype=np.float32)
    else:
        ht_pos = ht[ht > 0]
    ht_peak = float(np.max(ht_pos)) if ht_pos.size > 0 else 0.0
    legal_sub = (z_sub + ht_peak) <= (float(box_h) + 1e-6)
    return xs, ys, z_sub, legal_sub


def _principal_view_maps(env: Any, item_id: int, box_h: float, canvas_shape: tuple[int, int]) -> np.ndarray:
    # Paper-style principal views: front, rear, left, right, top, bottom.
    ori_top = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ori_front = np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64)
    ori_rear = np.array([3.0 * np.pi / 2.0, 0.0, 0.0], dtype=np.float64)
    ori_left = np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float64)
    ori_right = np.array([0.0, 3.0 * np.pi / 2.0, 0.0], dtype=np.float64)

    maps: list[np.ndarray] = []
    for ori in (ori_front, ori_rear, ori_left, ori_right):
        hm_top, _ = env.item_hm(item_id, ori)
        canvas = np.zeros(canvas_shape, dtype=np.float32)
        if hm_top.size > 0:
            hm_top_norm = np.clip(hm_top / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)
            canvas = _center_overlay(canvas, hm_top_norm)
        maps.append(canvas)

    top_hm, bottom_hm = env.item_hm(item_id, ori_top)

    top_canvas = np.zeros(canvas_shape, dtype=np.float32)
    if top_hm.size > 0:
        top_norm = np.clip(top_hm / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)
        top_canvas = _center_overlay(top_canvas, top_norm)

    bottom_canvas = np.zeros(canvas_shape, dtype=np.float32)
    if bottom_hm.size > 0:
        bottom_finite = bottom_hm.copy()
        bottom_finite[~np.isfinite(bottom_finite)] = 0.0
        bottom_norm = np.clip(bottom_finite / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)
        bottom_canvas = _center_overlay(bottom_canvas, bottom_norm)

    maps.extend([top_canvas, bottom_canvas])
    return np.stack(maps, axis=0).astype(np.float32)


def manager_candidates(
    env: Any,
    unpacked: list[int],
    max_objects_k: int = PAPER_MANAGER_TOP_K,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    if max_objects_k > 0 and len(unpacked) > max_objects_k:
        ordered = sorted(unpacked, key=lambda item_id: aabb_volume(env, item_id), reverse=True)[:max_objects_k]
    else:
        ordered = list(unpacked)

    box_hm = env.box_heightmap(copy=False)
    box_h = float(env.box_size[2])
    box_l, box_w = float(env.box_size[0]), float(env.box_size[1])
    box_norm = _normalize_hm(box_hm, box_h)

    ucount = len(ordered)
    fill_ratio = float(np.sum(box_hm) / max(1e-6, env.resolution * env.resolution * box_h))
    box_volume = float(box_l * box_w * box_h)

    cands: list[tuple[np.ndarray, np.ndarray, int]] = []
    for item_id in ordered:
        dx, dy, dz, volume = item_aabb_features(env, item_id)
        principal = _principal_view_maps(env, item_id, box_h=box_h, canvas_shape=box_norm.shape)
        map_state = np.concatenate([box_norm[None, ...], principal], axis=0).astype(np.float32)
        scalar_state = np.array(
            [
                dx / max(1e-6, box_l),
                dy / max(1e-6, box_w),
                dz / max(1e-6, box_h),
                volume / max(1e-6, box_volume),
                float(np.mean(box_norm)),
                float(np.max(box_norm)),
                fill_ratio,
                float(ucount) / 64.0,
            ],
            dtype=np.float32,
        )
        cands.append((map_state, scalar_state, int(item_id)))
    return cands


def worker_candidates(
    env: Any,
    item_id: int,
    grid_step: int,
    max_candidates: int,
    rng: np.random.Generator,
    orientation_step: float = PAPER_ORIENTATION_STEP,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    box_hm = env.box_heightmap(copy=False)
    box_h = float(env.box_size[2])
    box_norm = _normalize_hm(box_hm, box_h)

    orientations = orientation_grid(step=orientation_step)
    cands: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for ori in orientations:
        ht, hb = env.item_hm(item_id, ori)
        if ht.size == 0 or hb.size == 0:
            continue
        w, h = ht.shape
        if w > env.resolution or h > env.resolution:
            continue

        x_range = range(0, env.resolution - w + 1, grid_step)
        y_range = range(0, env.resolution - h + 1, grid_step)

        hb_finite = hb.copy()
        hb_finite[~np.isfinite(hb_finite)] = 0.0
        hb_norm = np.clip(hb_finite / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)

        for x in x_range:
            for y in y_range:
                z = float(np.max(box_hm[x : x + w, y : y + h] - hb))
                updated = np.maximum((ht > 0) * (ht + z), box_hm[x : x + w, y : y + h])
                if float(np.max(updated)) > box_h:
                    continue

                placed_top = np.zeros_like(box_norm, dtype=np.float32)
                local_support = np.zeros_like(box_norm, dtype=np.float32)
                placed_top[x : x + w, y : y + h] = np.clip((ht + z) / max(1e-6, box_h), 0.0, 1.0)
                local_support[x : x + w, y : y + h] = hb_norm

                map_state = np.stack([box_norm, placed_top, local_support], axis=0).astype(np.float32)

                roll, pitch, yaw = [float(v) for v in ori]
                scalar_state = np.array(
                    [
                        x / float(env.resolution),
                        y / float(env.resolution),
                        z / max(1e-6, box_h),
                        w / float(env.resolution),
                        h / float(env.resolution),
                        np.sin(roll),
                        np.cos(roll),
                        np.sin(pitch),
                        np.cos(pitch),
                        np.sin(yaw),
                        np.cos(yaw),
                    ],
                    dtype=np.float32,
                )

                action = np.array([roll, pitch, yaw, x, y, z], dtype=np.float64)
                cands.append((map_state, scalar_state, action))

    if max_candidates > 0 and len(cands) > max_candidates:
        idx = rng.choice(len(cands), size=max_candidates, replace=False)
        cands = [cands[int(i)] for i in idx]

    return cands


def worker_scalar_state(
    env: Any,
    x: int,
    y: int,
    z: float,
    w: int,
    h: int,
    ori: np.ndarray,
) -> np.ndarray:
    box_h = float(env.box_size[2])
    roll, pitch, yaw = [float(v) for v in ori]
    return np.array(
        [
            x / float(env.resolution),
            y / float(env.resolution),
            z / max(1e-6, box_h),
            w / float(env.resolution),
            h / float(env.resolution),
            np.sin(roll),
            np.cos(roll),
            np.sin(pitch),
            np.cos(pitch),
            np.sin(yaw),
            np.cos(yaw),
        ],
        dtype=np.float32,
    )


def worker_orientation_candidates(
    env: Any,
    item_id: int,
    grid_step: int,
    orientation_step: float = PAPER_ORIENTATION_STEP,
    use_cuda: bool = False,
) -> list[dict]:
    """Build per-orientation worker states for score-map planning.

    Returns dict entries with keys:
    - map_state: (3,H,W) float32 input for worker model
    - ori: (3,) roll/pitch/yaw
    - w, h: object footprint dims in heightmap cells
    - legal_mask: (H,W) float32, 1 for legal placements
    - z_map: (H,W) float32, placement z value for each legal cell
    """
    box_h = float(env.box_size[2])
    box_hm = None
    box_norm = None
    box_norm_t = None
    box_hm_t = None
    if use_cuda and hasattr(env, "box_heightmap_t"):
        box_hm_t = env.box_heightmap_t(copy=False)
        box_norm_t = torch.clamp(box_hm_t / max(1e-6, box_h), 0.0, 1.0).to(torch.float32)
    else:
        box_hm = env.box_heightmap(copy=False)
        box_norm = _normalize_hm(box_hm, box_h)
        if use_cuda:
            box_norm_t = torch.from_numpy(np.asarray(box_norm, dtype=np.float32)).to("cuda")
            box_hm_t = torch.from_numpy(np.asarray(box_hm, dtype=np.float32)).to("cuda")
    H = env.resolution
    W = env.resolution

    orientations = orientation_grid(step=orientation_step)
    out: list[dict] = []
    stride = max(1, int(grid_step))

    for ori in orientations:
        if use_cuda and hasattr(env, "item_hm_torch"):
            ht_t, hb_t = env.item_hm_torch(item_id, ori, device="cuda")
            w, h = int(ht_t.shape[0]), int(ht_t.shape[1])
        else:
            ht_t, hb_t = None, None
            ht, hb = env.item_hm(item_id, ori)
            if ht.size == 0 or hb.size == 0:
                continue
            w, h = ht.shape
        if w > H or h > W:
            continue

        if use_cuda and ht_t is not None and hb_t is not None:
            ht_norm_t = torch.clamp(ht_t / max(1e-6, box_h), 0.0, 1.0)
            hb_norm_t = torch.clamp(
                torch.nan_to_num(hb_t, nan=0.0, posinf=0.0, neginf=0.0) / max(1e-6, box_h),
                0.0,
                1.0,
            )
        else:
            ht_norm = np.clip(ht / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)
            hb_finite = hb.copy()
            hb_finite[~np.isfinite(hb_finite)] = 0.0
            hb_norm = np.clip(hb_finite / max(1e-6, box_h), 0.0, 1.0).astype(np.float32)

        xs, ys, z_sub, legal_sub = _compute_z_and_legal_maps(
            box_hm=box_hm,
            ht=None if (use_cuda and ht_t is not None) else ht,
            hb=None if (use_cuda and hb_t is not None) else hb,
            box_h=box_h,
            stride=stride,
            use_cuda=use_cuda,
            ht_t=ht_t,
            hb_t=hb_t,
            box_t=box_hm_t,
        )
        if (isinstance(z_sub, torch.Tensor) and int(z_sub.numel()) == 0) or (
            not isinstance(z_sub, torch.Tensor) and z_sub.size == 0
        ):
            continue

        if use_cuda:
            # Keep heavy candidate maps on GPU. Convert to CPU only for chosen actions.
            legal_sub_t = (
                legal_sub
                if isinstance(legal_sub, torch.Tensor)
                else torch.from_numpy(np.asarray(legal_sub, dtype=np.bool_)).to("cuda")
            )
            z_sub_t = (
                z_sub
                if isinstance(z_sub, torch.Tensor)
                else torch.from_numpy(np.asarray(z_sub, dtype=np.float32)).to("cuda")
            )

            item_top_t = torch.zeros_like(box_norm_t, dtype=torch.float32)
            item_bottom_t = torch.zeros_like(box_norm_t, dtype=torch.float32)
            x0 = (H - w) // 2
            y0 = (W - h) // 2
            if ht_t is not None and hb_t is not None:
                item_top_t[x0 : x0 + w, y0 : y0 + h] = ht_norm_t
                item_bottom_t[x0 : x0 + w, y0 : y0 + h] = hb_norm_t
            else:
                item_top_t[x0 : x0 + w, y0 : y0 + h] = torch.from_numpy(np.asarray(ht_norm, dtype=np.float32)).to("cuda")
                item_bottom_t[x0 : x0 + w, y0 : y0 + h] = torch.from_numpy(np.asarray(hb_norm, dtype=np.float32)).to("cuda")
            map_state_t = torch.stack([box_norm_t, item_top_t, item_bottom_t], dim=0)

            legal_mask_t = torch.zeros((H, W), dtype=torch.bool, device="cuda")
            z_map_t = torch.zeros((H, W), dtype=torch.float32, device="cuda")
            xs_t = torch.from_numpy(np.asarray(xs, dtype=np.int64)).to("cuda")
            ys_t = torch.from_numpy(np.asarray(ys, dtype=np.int64)).to("cuda")
            legal_mask_t[torch.meshgrid(xs_t, ys_t, indexing="ij")] = legal_sub_t
            z_map_t[torch.meshgrid(xs_t, ys_t, indexing="ij")] = torch.where(
                legal_sub_t,
                z_sub_t,
                torch.zeros_like(z_sub_t),
            )

            out.append(
                {
                    "map_state_t": map_state_t,
                    "ori": np.asarray(ori, dtype=np.float64),
                    "ori_t": torch.from_numpy(np.asarray(ori, dtype=np.float32)).to("cuda"),
                    "w": int(w),
                    "h": int(h),
                    "legal_mask_t": legal_mask_t,
                    "z_map_t": z_map_t,
                }
            )
        else:
            item_top = _center_overlay(np.zeros_like(box_norm, dtype=np.float32), ht_norm)
            item_bottom = _center_overlay(np.zeros_like(box_norm, dtype=np.float32), hb_norm)
            map_state = np.stack([box_norm, item_top, item_bottom], axis=0).astype(np.float32)

            legal_mask = np.zeros((H, W), dtype=np.float32)
            z_map = np.zeros((H, W), dtype=np.float32)
            legal_mask[np.ix_(xs, ys)] = legal_sub.astype(np.float32)
            z_map[np.ix_(xs, ys)] = np.where(legal_sub, z_sub, 0.0).astype(np.float32)

            if float(np.max(legal_mask)) <= 0.0:
                continue
            out.append(
                {
                    "map_state": map_state,
                    "ori": np.asarray(ori, dtype=np.float64),
                    "w": int(w),
                    "h": int(h),
                    "legal_mask": legal_mask,
                    "z_map": z_map,
                }
            )

    if use_cuda and len(out) > 0:
        has_legal_t = torch.stack([torch.any(c["legal_mask_t"]) for c in out], dim=0)
        keep_t = torch.nonzero(has_legal_t, as_tuple=False).squeeze(-1)
        if int(keep_t.numel()) == 0:
            return []
        keep_idx = keep_t.detach().cpu().tolist()
        out = [out[int(i)] for i in keep_idx]

        # Cache batched tensors once to avoid repeated Python-level stacking in hot paths.
        batch_cache = {
            "maps_t": torch.stack([c["map_state_t"] for c in out], dim=0),
            "legal_masks_t": torch.stack([c["legal_mask_t"] for c in out], dim=0),
            "z_maps_t": torch.stack([c["z_map_t"] for c in out], dim=0),
            "ori_t": torch.stack([c["ori_t"] for c in out], dim=0),
        }
        for idx, c in enumerate(out):
            c["_batch_cache"] = batch_cache
            c["_batch_idx"] = int(idx)

    return out
