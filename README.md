# IrregularPacking_DRL

Hierarchical deep-RL planner for **irregular 3D object packing**, adapted from Huang et al., *"Planning Irregular Object Packing via Hierarchical Reinforcement Learning"* (IEEE RA-L 2023, [DOI 10.1109/LRA.2022.3222996](https://doi.org/10.1109/LRA.2022.3222996)). The paper is included as `Paper.pdf` and the implementation contract is in `PAPER_SPEC.md`.

The original paper targets a UR5 robot packing groceries into a box and uses PyBullet for both perception and stability checks. **This repository drops the robot and the dynamic simulator** and treats packing as a pure geometry problem, so it can be used for **nesting parts in a 3D-printing build volume** (e.g. SLS, where the powder bed supports every part and stability is not a concern).

## What the planner does

For each episode the agent is given:

- a fixed-size container (build volume), represented by a top-down heightmap
- a set of unpacked parts, each represented by 3D meshes (URDF + OBJ)

Two networks decide what to pack and where, one part at a time:

- **Manager** (ResNet18 + 3-layer FC head) picks the next part from the unpacked set, using the box top-view plus six principal-view heightmaps (front, rear, left, right, top, bottom) of each candidate.
- **Worker** (14-layer U-Net) predicts an `(x, y, roll, pitch, yaw)` placement for the chosen part. `z` is computed analytically from the lowest collision-free drop onto the current heightmap (paper Eq. 1). Illegal positions are hard-masked out of the score map.

Both networks are trained with a hierarchical Double-DQN in two stages (worker pre-training, then joint training with a 4:1 update ratio), matching the paper's protocol. See `PAPER_SPEC.md` for the full reproduction checklist.

### Deviations from the paper

- **No PyBullet, no robot.** All collision and feasibility checks are done analytically on heightmaps. The geometry-only environment lives in `packing/env_packing.py`.
- **No stability term in the reward.** The paper uses `J = αC + βP + γS` with γ for placement stability. Here `γ = 0` because the SLS powder bed supports every part. See `SLS_SIMPLIFICATION_PLAN.md`.
- **Per-step shaped reward** instead of pure ΔJ. The default reward gives one fixed bonus per part successfully placed; optional Δcompactness / Δpyramidality / height-growth shaping terms are wired up but defaulted to zero. The objective metrics themselves use mesh volume (signed-tetrahedron integration) rather than AABB volume.
- **Build volume defaults to a 25.6 cm cube** in `scripts/train_hrl_packing.py`, configurable via `--box_size W D H`. The paper used 40 × 40 × 30 cm.

Everything else (top-K=20 sequence search, π/2 orientation grid, 200×200 heightmap resolution, Adam batch=128, LR 1e-3 → 1e-4 at joint phase, 0.8×–1.2× scale augmentation, 5,000-combination episode pool) follows the paper.

## Repository layout

```
packing/                   # active code
  env_packing.py           # geometry-only PackingEnv (mesh I/O, heightmaps, Eq. 1 drop)
  state.py                 # manager/worker state builders, orientation grid, top-K filter
  models_manager.py        # ResNet18 + scalar MLP + 3-layer FC head
  models_worker.py         # 14-layer U-Net score-map + scalar branch
  reward.py                # compactness / pyramidality / shaped per-step reward
  replay.py                # CPU- or GPU-backed replay buffers (HierarchicalReplay)
  agent_hrl.py             # Double-DQN updates, target nets, epsilon schedules
scripts/
  train_hrl_packing.py     # main two-stage trainer (manager + worker)
  train_packing.py         # worker-only trainer (simpler entry point)
  eval_packing.py          # inference-only evaluator (heuristic or learned policy)
  setup_recommended.ps1    # Windows setup helper
setup_paperspace.sh        # Paperspace setup helper
pybullet-object-models-master/   # YCB URDF+OBJ catalog (data only; not imported)
logs/                      # sample evaluation outputs from a previously trained ckpt
Paper.pdf                  # source paper
PAPER_SPEC.md              # reproduction contract derived from the paper
HRL_REFACTOR_PLAN.md       # design notes for the refactor away from the robot loop
SLS_SIMPLIFICATION_PLAN.md # rationale for dropping PyBullet/stability for SLS
FULL_GPU_ROLLOUT_PLAN.md   # notes on the GPU-native rollout/env path
GPU_SYNC_OPTIMIZATION_PLAN.md
pyramidality_example.py    # voxel-grid reference implementation of pyramidality
trash/                     # legacy push/grasp fork files, kept for reference, not used
```

## Installation

Python 3.10+ and PyTorch. CUDA is optional but strongly recommended.

```bash
pip install -r requirements.txt
```

For a CUDA build of PyTorch, follow the [PyTorch install matrix](https://pytorch.org/get-started/locally/) instead of relying on the default wheel.

## Quick start

Train the full hierarchical agent (manager + worker) on the bundled YCB catalog:

```bash
python scripts/train_hrl_packing.py \
  --obj_dir pybullet-object-models-master \
  --episodes 300 --stage1_episodes 100 \
  --num_objects 50 --resolution 200 \
  --box_size 0.256 0.256 0.256 \
  --save_path logs/packing_hrl.pt
```

Train only the worker with a heuristic (bbox-volume) sequence:

```bash
python scripts/train_packing.py --obj_dir pybullet-object-models-master
```

Evaluate a trained checkpoint:

```bash
python scripts/eval_packing.py \
  --obj_dir pybullet-object-models-master \
  --model_path logs/packing_hrl.pt \
  --episodes 5 --num_objects 10
```

Evaluate the heuristic baseline (no model) on the same task:

```bash
python scripts/eval_packing.py --obj_dir pybullet-object-models-master --episodes 5
```

Important flags for `scripts/train_hrl_packing.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--box_size W D H` | `0.256 0.256 0.256` | Build volume in metres |
| `--resolution` | `200` | Heightmap grid (paper: 200) |
| `--num_objects` | `50` | Parts sampled per episode |
| `--manager_top_k` | `20` | Sequence-search size (paper: 20) |
| `--orientation_step` | `π/2` | Roll/pitch/yaw discretization |
| `--grid_step` | `4` | Stride over (x, y) candidate placements |
| `--stage1_episodes` | `100` | Worker pre-training episodes before joint phase |
| `--manager_update_interval_epochs` | `4` | Joint-phase ratio (paper: 4:1) |
| `--batch_size` | `128` | Paper default |
| `--manager_lr` / `--worker_lr` | `1e-3` | Stage-1 learning rate |
| `--stage2_lr` | `1e-4` | Joint-phase learning rate |
| `--cpu` / `--cpu_replay` | off / on | Force CPU training / keep replay on host RAM |
| `--num_workers` | `1` | Multi-process rollout (CPU only) |
| `--resume PATH` | — | Resume from a `--checkpoint_path` snapshot |

## Object catalog

The included `pybullet-object-models-master/` folder contains ~10 YCB items (URDF + OBJ). The environment loads any URDF tree under `--obj_dir` directly — no Python install is required. You can drop your own STL/OBJ-based URDFs into a parallel directory and point `--obj_dir` at it, or add an `objects.csv` with a `dir` column listing relative URDF paths for explicit ordering.

To use your own 3D-printing parts, convert each STL to OBJ, wrap it in a minimal URDF that references the OBJ in its `<collision><geometry><mesh filename="..."/>` element, and place it under the object root.

## Notes

- `trash/` holds files inherited from the upstream push/grasp fork (`main.py`, `models.py`, `trainer.py`, `robot.py`, `env.py`, `evaluate.py`, `logger.py`, `utils.py`, `heuristics_HM.py`). They form a closed import cluster and are not used by any active module. Delete the folder if you don't need it.
- `Paper.pdf` is © IEEE 2023, included for reference under personal-use rights.
