# SLS Simulation Simplification Plan

Goal: make training/evaluation as time-efficient as possible for SLS-style packing, where gravity and dynamic settling are not required.

## Assumptions
- No gravity-driven stability behavior is needed.
- We care about geometric feasibility and packing quality, not rigid-body dynamics realism.
- This path becomes the default behavior (no extra mode flag).

## Proposed Changes (In Order)

## 1) Remove dynamic settling loop
- Change: eliminate per-placement simulation stepping (`p.stepSimulation()` settle loop) from `pack_item`.
- Why: this is currently one of the largest per-step costs.
- Expected impact: high speedup, low implementation risk.

## 2) Replace physics-stability check with geometric validity
- Change: stop using post-simulation pose drift/orientation drift as “stable”.
- New rule: if placement is inside bounds, collision-free by heightmap constraints, and within box height, treat as valid/stable.
- Why: pose drift checks are dynamics-based and not meaningful for SLS static packing.
- Expected impact: moderate speedup, simplifies reward/stability semantics.

## 3) Use direct geometric placement (heightmap-first), not body dynamics
- Change: make `env.step` update internal packing state from geometric placement result directly.
- Avoid relying on dynamic simulation outcome to decide success/failure.
- Why: keeps state transitions deterministic and aligned with static packing objective.
- Expected impact: high consistency and speed.

## 4) Cache per-object/per-orientation heightmaps within an episode
- Change: cache `item_hm(item_id, orientation)` results and reuse across candidate generation.
- Clear cache on episode reset.
- Why: repeated `item_hm` queries are expensive and heavily repeated.
- Expected impact: high speedup, especially at higher resolution and denser orientation/grid search.

## 5) Minimize PyBullet scene overhead
- Change: remove non-essential simulation elements for this workflow where possible (e.g., dynamic dependencies not used by geometric checks).
- Keep only what is required for mesh loading and shape querying.
- Why: reduces setup and per-step overhead.
- Expected impact: moderate speedup.

## 6) Optional: reduce repeated conversions/allocations in hot loops
- Change: avoid repeated array allocations/copies in candidate generation and action scoring paths.
- Why: reduces Python/NumPy overhead.
- Expected impact: moderate speedup.

## Validation Plan
- Baseline and compare:
  - Mean episode wall time
  - Mean step wall time
  - Throughput (episodes/hour)
  - Reward trend stability
- Confirm behavior remains coherent:
  - No invalid placements outside box
  - No overflow above box height
  - Monotonic progression of packed/unpacked sets

## Recommended Implementation Sequence
1. Change 1 (remove settling loop)
2. Change 2 (geometric stability rule)
3. Change 4 (heightmap cache)
4. Change 3 (direct geometric transition path cleanup)
5. Change 5/6 (overhead reductions)

## Notes
- This plan intentionally prioritizes throughput over dynamic realism.
- Since this becomes default behavior, all train/eval scripts should inherit the same simplified environment semantics.
