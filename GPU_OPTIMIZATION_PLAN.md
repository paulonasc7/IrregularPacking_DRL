# GPU Optimization Plan (No Training-Logic Changes)

This plan targets the three current CPU hotspots and moves heavy tensor math to GPU.

## Important Constraint

These changes are implementation optimizations only:
- No reward-definition changes
- No policy/objective changes
- No action-space changes
- No episode-termination logic changes

Expected differences should be limited to:
- Runtime speed
- Minor floating-point-level numeric drift

---

## 1) Move `_compute_z_and_legal_maps` to GPU

**Current hotspot**
- `packing/state.py:_compute_z_and_legal_maps`
- Dominant CPU cost in profiles.

**What to change**
- Implement a torch path that computes:
  - sampled sliding windows over `box_hm`
  - `z_sub = max(window - hb)` per sampled placement
  - legality mask from `(z_sub + ht_peak) <= box_h`
- Use GPU tensors when device is CUDA.
- Keep current NumPy path as fallback for CPU mode.

**Why this helps**
- This function is called many times per step and is the largest cumulative cost.

**Validation**
- For a fixed input batch, compare CPU vs GPU outputs:
  - same shapes
  - close values (`allclose`)
  - same boolean legality pattern (or tiny tolerance-driven edge differences)

---

## 2) Keep `worker_orientation_candidates` map construction on GPU

**Current hotspot**
- `packing/state.py:worker_orientation_candidates`
- Repeated CPU allocations/copies for `legal_mask`, `z_map`, and map tensors.

**What to change**
- Build `box_norm`, `item_top`, `item_bottom`, masks, and z maps as torch tensors on GPU.
- Avoid converting back to NumPy until strictly needed (ideally only selected action details).
- Reuse preallocated tensors where possible.

**Why this helps**
- Reduces host memory churn and repeated CPU array materialization.

**Validation**
- Candidate count unchanged for the same seed/config.
- Chosen greedy action matches baseline in deterministic runs (or near-equivalent under tie cases).

---

## 3) GPU tensorization/caching for `item_hm` transform path

**Current hotspot (secondary)**
- `packing/env_packing.py:item_hm` and orientation transform helpers.

**What to change**
- Tensorize orientation transform and projection operations (torch on CUDA).
- Keep and reuse cached rotated/top-bottom maps for repeated orientation queries.
- Preserve exact cache keys and API behavior.

**Why this helps**
- Lower per-orientation prep overhead, especially when many orientation queries are made per step.

**Validation**
- `item_hm` output shapes unchanged.
- Top/bottom maps numerically close to baseline.
- Placement validity decisions remain consistent in test episodes.

---

## Rollout Strategy

1. Implement item 1 first (highest impact).
2. Benchmark before/after with the same short-run command and seed.
3. Implement item 2.
4. Benchmark again.
5. Implement item 3.
6. Final benchmark + quick training sanity check.

## Suggested Benchmark Command

```powershell
python scripts/train_hrl_packing.py --episodes 1 --max_steps 10 --num_objects 20 --resolution 200 --grid_step 4 --num_workers 1 --log_every 1
```

