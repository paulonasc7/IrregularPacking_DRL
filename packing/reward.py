from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    # --- Objective weights (used in objective_value) ---
    # Both metrics are in [0, 1]; equal weight recommended for SLS.
    compactness: float = 0.5   # packed_vol / (XY_bbox_area * max_height)
    pyramidality: float = 0.3  # packed_vol / occupied_volume (penalises voids)

    # --- Per-step shaping weights (used in transition_reward) ---
    # delta_* terms telescope (sum = final - initial), so also add step_* terms
    # for a dense non-telescoping signal that rewards high density throughout.
    delta_compactness_bonus: float = 0.40   # reward per-step compactness improvement
    delta_pyramidality_bonus: float = 0.40  # reward per-step pyramidality improvement
    step_density_bonus: float = 0.20        # reward current objective value each step
    height_growth_penalty: float = 0.10     # penalise height growth (lower build = less print time)

    # --- Removed for SLS (kept at 0 for API compatibility) ---
    stability: float = 0.0        # powder supports all parts - irrelevant
    success_bonus: float = 0.0    # stable=True always for legal placements
    invalid_penalty: float = 0.0  # worker never selects illegal placements
    step_penalty: float = 0.0
    step_objective_bonus: float = 0.0   # superseded by step_density_bonus
    compact_gain_bonus: float = 0.0     # superseded by delta_compactness_bonus
    pyramid_gain_bonus: float = 0.0     # superseded by delta_pyramidality_bonus
    simple_mode: bool = False  # unused; single clean code path now


def _bbox_fill_ratio(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    xs, ys = np.where(mask)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    bbox_area = max(1, w * h)
    return float(np.sum(mask) / bbox_area)


def _heightmap_metrics(
    heightmap: np.ndarray,
    box_size: tuple[float, float, float],
    packed_parts_volume: float,
) -> tuple[float, float, float, float]:
    """Return (occupied_volume, max_height, compactness, pyramidality).

    compactness:
      3D envelope fill inside the occupied XY AABB and current max height.
      This better penalizes scattered placements than using full-box area.

    pyramidality:
      Multi-level slice compactness: average 2D fill ratio of occupied masks at
      increasing height thresholds. Higher values indicate denser, less ragged
      layering across heights.
    """
    hm = np.asarray(heightmap, dtype=np.float64)
    packed_parts_volume = max(0.0, float(packed_parts_volume))
    if hm.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    box_l, box_w, box_h = [float(v) for v in box_size]
    h, w = hm.shape
    cell_area = (box_l / max(1, w)) * (box_w / max(1, h))
    eps = 1e-8

    hm_pos = np.maximum(hm, 0.0)
    occupied_volume = float(np.sum(hm_pos) * cell_area)
    max_h = float(np.max(hm))
    if max_h <= eps:
        return occupied_volume, max_h, 0.0, 0.0

    occ_mask = hm_pos > eps
    if np.any(occ_mask):
        xs, ys = np.where(occ_mask)
        w_occ = int(xs.max() - xs.min() + 1)
        h_occ = int(ys.max() - ys.min() + 1)
        occ_bbox_area = float(max(1, w_occ * h_occ) * cell_area)
    else:
        occ_bbox_area = 0.0

    envelope_volume = occ_bbox_area * max_h  # bounding box envelope
    compactness = packed_parts_volume / max(eps, envelope_volume)
    compactness = float(np.clip(compactness, 0.0, 1.0))

    # Pyramidality per paper: P = mesh_vol / occupied_vol (projection to floor).
    # P = 1.0 means perfect stacking (no vertical gaps).
    # P < 1.0 means gaps/voids exist under or between parts.
    pyramidality_raw = packed_parts_volume / max(eps, occupied_volume)
    pyramidality = float(np.clip(pyramidality_raw, 0.0, 1.0))

    return occupied_volume, max_h, compactness, pyramidality


def objective_value(
    heightmap: np.ndarray,
    stable: bool,
    box_size: tuple[float, float, float],
    packed_parts_volume: float,
    weights: RewardWeights,
) -> tuple[float, dict]:
    occupied_volume, max_h, compactness, pyramidality = _heightmap_metrics(
        heightmap=heightmap,
        box_size=box_size,
        packed_parts_volume=packed_parts_volume,
    )
    # Stability intentionally excluded: SLS powder supports all parts.
    objective = (
        weights.compactness * compactness
        + weights.pyramidality * pyramidality
    )
    info = {
        "occupied_volume": float(occupied_volume),
        "packed_parts_volume": float(max(0.0, packed_parts_volume)),
        "max_height": float(max_h),
        "compactness": float(compactness),
        "pyramidality": float(pyramidality),
        "objective": float(objective),
    }
    return float(objective), info


def transition_reward(
    prev_heightmap: np.ndarray,
    next_heightmap: np.ndarray,
    prev_packed_parts_volume: float,
    next_packed_parts_volume: float,
    stable: bool,
    valid_action: bool,
    box_size: tuple[float, float, float],
    weights: RewardWeights,
) -> tuple[float, dict]:
    prev_obj, prev_info = objective_value(
        prev_heightmap,
        stable=False,
        box_size=box_size,
        packed_parts_volume=prev_packed_parts_volume,
        weights=weights,
    )
    next_obj, next_info = objective_value(
        next_heightmap,
        stable=False,  # stable excluded from objective for SLS
        box_size=box_size,
        packed_parts_volume=next_packed_parts_volume,
        weights=weights,
    )

    # Per-step improvement in each metric (clamped: only reward gains, not losses,
    # since a single bad step can temporarily lower metrics before recovery).
    delta_compactness = max(0.0, float(next_info["compactness"]) - float(prev_info["compactness"]))
    delta_pyramidality = max(0.0, float(next_info["pyramidality"]) - float(prev_info["pyramidality"]))

    # Non-telescoping dense signal: reward the current packing quality each step.
    # This differentiates policies that achieve high density early vs. late.
    step_density = float(next_obj)

    # Height growth penalty: lower build height = less SLS print time.
    box_h = max(1e-8, float(box_size[2]))
    prev_h = max(0.0, float(prev_info["max_height"]))
    next_h = max(0.0, float(next_info["max_height"]))
    height_growth = max(0.0, (next_h - prev_h) / box_h)

    reward = (
        float(weights.delta_compactness_bonus) * delta_compactness
        + float(weights.delta_pyramidality_bonus) * delta_pyramidality
        + float(weights.step_density_bonus) * step_density
        - float(weights.height_growth_penalty) * height_growth
    )

    info = {
        "prev_objective": float(prev_obj),
        "next_objective": float(next_obj),
        "delta_compactness": float(delta_compactness),
        "delta_pyramidality": float(delta_pyramidality),
        "step_density": float(step_density),
        "height_growth": float(height_growth),
        "compact_gain": float(delta_compactness),   # kept for logging compatibility
        "pyramid_gain": float(delta_pyramidality),  # kept for logging compatibility
        "prev_compactness": float(prev_info["compactness"]),
        "next_compactness": float(next_info["compactness"]),
        "prev_pyramidality": float(prev_info["pyramidality"]),
        "next_pyramidality": float(next_info["pyramidality"]),
        "shaped_reward": float(reward),
        "valid_action": bool(valid_action),
    }
    return float(reward), info
