# GPU Sync/Transfer Optimization Plan (No Logic Changes)

This plan targets overhead from GPU<->CPU sync points and repeated tensor conversion.

## Scope Guarantee

These are implementation-level performance optimizations only.

- No reward/objective changes
- No policy update rule changes
- No action-space changes
- No episode/termination rule changes
- No intended behavior changes (except tiny floating-point drift)

---

## 1) Keep Decision Flow on GPU (Minimize `.item()` usage)

**Goal**
- Avoid repeated scalar extraction from CUDA tensors inside loops.

**Changes**
- In worker selection/max-Q paths, keep argmax/mask computations fully on GPU.
- Extract only final needed scalar indices once per decision step.

**Why**
- `Tensor.item()` introduces synchronization barriers and stalls GPU pipelines when overused.

**Validation**
- Same selected action as baseline in deterministic tests (allowing tie-break differences).
- Same episode-level reward trend under same seed/config.

---

## 2) Reduce Frequency of CPU Scalar Reads

**Goal**
- Replace many small syncs with one or few syncs.

**Changes**
- Refactor per-candidate scalar reads into batched tensor operations.
- Read back only compact final results (e.g., one argmax index tuple).

**Why**
- Fewer sync points generally improves throughput and utilization.

**Validation**
- Count/log number of `.item()` calls in hot paths before vs after.
- Verify no change in success/reward distribution on short benchmark runs.

---

## 3) Reuse Persistent CUDA Tensors/Buffers

**Goal**
- Avoid repeated `torch.as_tensor(...)` and allocations in inner loops.

**Changes**
- Precreate and reuse per-step/per-orientation tensors where shapes are stable.
- Cache static tensors (box maps, masks, index tensors) as feasible.

**Why**
- Allocation and conversion overhead accumulates quickly in candidate-heavy loops.

**Validation**
- Reduced `torch.as_tensor` and allocation counts in profile.
- No functional output differences beyond tiny numerical noise.

---

## 4) Eliminate Unnecessary Host-Device Copies

**Goal**
- Prevent avoidable `.cpu().numpy()` conversions in hot paths.

**Changes**
- Keep intermediate maps/masks/scores on CUDA.
- Convert to CPU only where required by existing env/replay API boundaries.

**Why**
- Host-device transfer can dominate runtime even if model inference is fast.

**Validation**
- Profile shows reduced time in tensor `cpu()` and conversion calls.
- Training outputs remain consistent.

---

## 5) Increase Kernel Batching (Bigger Ops, Fewer Calls)

**Goal**
- Execute fewer, larger GPU kernels instead of many tiny ones.

**Changes**
- Merge repeated small operations into batched tensor operations where possible.
- Keep orientation/candidate computations vectorized at higher granularity.

**Why**
- Better GPU occupancy and lower launch overhead.

**Validation**
- Lower kernel-launch overhead in runtime profile.
- Equivalent candidate legality and action-selection behavior.

---

## Execution Order (Recommended)

1. Step 1
2. Step 2
3. Step 3
4. Step 4
5. Step 5

This order usually gives best speedup per engineering effort.

## Benchmark Command

```powershell
python scripts/train_hrl_packing.py --episodes 1 --max_steps 10 --num_objects 20 --resolution 200 --grid_step 4 --num_workers 1 --log_every 1
```

