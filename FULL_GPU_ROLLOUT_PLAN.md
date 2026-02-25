# Full GPU Rollout Plan (Single-Process, Minimal CPU Boundaries)

Goal: keep much more of rollout/training dataflow on GPU while keeping RL logic unchanged.

## Scope Guarantee

These changes are performance-oriented implementation refactors only.

- No reward/objective definition changes
- No policy architecture intent changes
- No action-space changes
- No episode/termination semantics changes
- Expected differences limited to runtime + tiny floating-point drift

---

## 1) GPU-Native Environment State

**Current gap**
- `box_heightmap` and much step math still rely on NumPy arrays.

**Change**
- Add/enable CUDA tensor representation for env state (`box_heightmap_t`).
- Run heightmap update/validation ops in torch on CUDA when GPU mode is enabled.
- Keep CPU view only for logging/debug API compatibility.

**Validation**
- Same step acceptance/rejection decisions for fixed seeds.
- Same packed/unpacked transitions and done conditions.

---

## 2) Fully GPU `item_hm` Path

**Current gap**
- `item_hm` source pipeline originates in NumPy geometry operations.

**Change**
- Move item projection/rotation/aggregation to CUDA tensors.
- Cache orientation-specific `ht/hb` on GPU per item/scale/orientation key.
- Avoid NumPy fallback in the rollout hot path.

**Validation**
- Shape parity with current `item_hm`.
- Numerical closeness of `ht/hb` maps and legality decisions.

---

## 3) End-to-End GPU Action Pipeline

**Current gap**
- Most scoring is on GPU, but some conversions still happen.

**Change**
- Keep candidate maps, legality masks, z-maps, and argmax entirely on GPU.
- Extract only minimal action tuple if env step still requires Python boundary.
- If env step is GPU-native (Step 1), keep action + step fully tensorized.

**Validation**
- Same selected actions in deterministic runs (allow tie-break tolerance).
- Same reward trend under fixed seeds.

---

## 4) GPU Replay Staging

**Current gap**
- Transition packaging and copies still frequently hit CPU/NumPy.

**Change**
- Stage transitions as GPU tensors during rollout.
- Use batched transfer or pinned-memory staging only where optimizer/replay requires CPU.
- Reduce per-step tensor-to-NumPy copies.

**Validation**
- Replay sizes/counts unchanged.
- Loss curves and update frequencies unchanged.

---

## 5) Reduce Python Orchestration in Hot Loop

**Current gap**
- Candidate bookkeeping still uses many Python dict/list operations.

**Change**
- Replace per-candidate Python loops with batched tensor indexing where feasible.
- Keep metadata compact (tensor indices instead of large dict payloads).
- Avoid repeated per-step object allocations in hot paths.

**Validation**
- Same candidate counts and valid-action masks for fixed seeds.
- No behavioral changes in rollout outputs.

---

## 6) Lock to Single-Process GPU Path (`num_workers=1`)

**Current gap**
- Multi-process adds serialization/duplication overhead and complicates persistent GPU state.

**Change**
- Define and optimize a primary single-process CUDA training path.
- Keep multiprocessing path optional/secondary.

**Validation**
- Benchmark `num_workers=1` throughput and stability.
- Confirm deterministic behavior remains acceptable.

**Status**
- Implemented in `scripts/train_hrl_packing.py`:
  - CUDA + `num_workers>1` now defaults back to `num_workers=1` (primary path).
  - Secondary multiprocessing path remains available via `--allow_secondary_multiprocess`.

---

## Benchmark Protocol

Use the same config before/after each step:

```powershell
python scripts/train_hrl_packing.py --episodes 1 --max_steps 10 --num_objects 20 --resolution 200 --grid_step 4 --num_workers 1 --log_every 1
```

Track:
- `elapsed_sec`
- GPU utilization/memory (`nvidia-smi -l 1`)
- reward/success consistency

---

## Recommended Implementation Order

1. Step 1 (env state on GPU)
2. Step 2 (`item_hm` fully GPU)
3. Step 3 (end-to-end GPU action path)
4. Step 4 (replay staging)
5. Step 5 (Python hot-loop reduction)
6. Step 6 (single-process as primary path)
