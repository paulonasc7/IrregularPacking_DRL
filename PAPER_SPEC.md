# PAPER_SPEC.md

Source paper: `Paper.pdf` ("Planning Irregular Object Packing via Hierarchical Reinforcement Learning", IEEE RA-L 2023, DOI: `10.1109/LRA.2022.3222996`).

This document is the implementation contract for exact paper reproduction.

## 1. Task Definition

Goal: iteratively choose object sequence and placement to maximize packing performance in a fixed-size box.

- Sequence planning module (manager): select next object.
- Placement planning module (worker): predict `(X, Y, phi, theta, psi)`.
- `Z` is not predicted directly; compute it by lowest collision-free placement under gravity (Eq. 1).

Stop condition:
- All objects packed, or
- no feasible space remains for unpacked objects.

## 2. State/Action (Exact Paper)

### 2.1 Manager (Sequence Planning)

State `s_m`:
- Current box top-view heightmap `Hc`.
- For each unpacked object: six principal-view heightmaps (front, rear, left, right, top, bottom).
- Principal views are taken from stable poses; heightmap zero-plane in each view is the opposite plane of object bounding box.

Action `a_m`:
- Choose one object from unpacked set.
- Paper note for large candidate sets: restrict search to top-`K` unpacked objects by bounding-box volume.
- Default in paper experiments: `K = 20`.

Manager scoring behavior:
- Concatenate each object's six-view maps with box map to form per-object feature input.
- Use CNN features and per-object score prediction.
- Packed/already-selected objects are represented by zero feature maps for dimension consistency.

### 2.2 Worker (Placement Planning)

State `s_w`:
- Box top-view heightmap `Hc`.
- Selected object top-view and bottom-view heightmaps under candidate orientation.

Action `a_w`:
- Predict horizontal location `(X, Y)` and orientation `(phi, theta, psi)`.
- Only top-down placement trajectories are allowed.

Orientation discretization:
- Roll/pitch grid: `Orp = {(phi_i, theta_i)}` over `[0, 2pi)`.
- Yaw grid: `Oy = {0, psi_1, ..., psi_m}` with equal interval over `[0, 2pi)`.
- Paper default search interval for roll/pitch/yaw: `pi/2`.

Per-orientation score maps:
- For each `(phi_i, theta_i, psi_j)`, predict score matrix `W_ij` with same spatial size as `Hc`.
- `W_ij[x, y]` = score of placing object center at `(x, y)` with that orientation.
- Illegal positions must be masked to zero:
  - collision with box boundary/margins,
  - placement causing height overflow beyond box height.

Vertical placement (Eq. 1):
- Compute `z` from bottom-view object heightmap and current box heightmap:
- `z = max_s max_t ( Hc[x+s, y+t] - H_b^{ij}[s,t] )`
- with `s,t` ranges over footprint extents (`w,h`) as in paper Eq. (1).

## 3. Objective/Reward (Exact Paper)

State objective:
- `J(s_t) = alpha * C + beta * P + gamma * S`  (Eq. 2)
- Expanded (Eq. 3):
  - `C = (sum_{i=1..t} V_i) / (L * W * h_t)`
  - `P = (sum_{i=1..t} V_i) / V_p^t`
  - `S in {0,1}` from placement stability check.

Definitions:
- `V_i`: volume of packed object `i`.
- `L,W`: box length/width.
- `h_t`: current max packed height (largest value in box heightmap).
- `V_p^t`: projected volume of packed objects to bottom (from box heightmap sum).
- `S = 1` if stable placement, else `0`.

Per-step reward:
- `r(s_t, a_t) = J(s_{t+1}) - J(s_t)`  (Eq. 4)

Paper default weights:
- `alpha = 0.75`
- `beta = 0.25`
- `gamma = 0.25`

Stability thresholds (paper defaults):
- position difference < `2 cm`
- orientation difference < `pi/6`

## 4. Training Protocol (Exact Paper)

Two-stage hierarchical Q-learning:

Stage 1 (worker pretraining):
- Train placement planning module only.
- Use heuristic sequence sorted by bounding-box volume.

Stage 2 (joint training):
- Replace heuristic sequence by learned manager.
- Jointly optimize manager + worker.
- Different time scales during joint training:
  - update worker every epoch,
  - update manager every 4 epochs.

Network choices in paper:
- Manager backbone: ResNet18 + 3-layer FC head for object selection scores.
- Worker network: U-Net (14 layers) producing placement score map.

Optimizer/hyperparams:
- Adam, batch size `128`.
- Learning rate:
  - `1e-3` in stage 1,
  - `1e-4` in stage 2.

## 5. Environment/Data/Eval Defaults (Paper)

Simulation platform:
- PyBullet.

Default box and map resolution:
- Box size `40cm x 40cm x 30cm`.
- Heightmap resolution `200 x 200` (2mm per pixel).
- Object heightmaps scanned at same resolution.
- HM heuristic baseline uses downsampled `50 x 50` box map.

Datasets:
- YCB + OCRTOC.
- 121 object categories.
- Per packing case: 50 randomly selected instances.
- Random scale augmentation: `0.8x` to `1.2x`.
- Dataset splits in paper:
  - 5,000 object combinations for training,
  - 2,000 combinations for test.

Metrics reported:
- Compactness.
- Pyramidality.
- Stability.
- Average packed object count.
- Latency per object.

## 6. Reproduction Compliance Checklist

A run is "paper-faithful" only if all are true:

- [ ] Manager input uses six principal-view object heightmaps + box map (not reduced proxy features).
- [ ] Worker input uses top+bottom object maps + box map; outputs per-orientation score maps.
- [ ] Orientation search uses paper discretization with default `pi/2` intervals.
- [ ] Illegal placements are hard-masked to zero in score maps.
- [ ] `z` is computed by Eq. (1) style max over footprint support.
- [ ] Reward is exactly delta objective `J(s_{t+1}) - J(s_t)` with paper `C/P/S` definitions.
- [ ] Stage-1 worker pretraining uses bbox-volume sequence heuristic.
- [ ] Stage-2 joint training uses manager+worker with worker:manager update ratio `4:1` by epoch.
- [ ] Manager model family matches paper (ResNet18 + FC head).
- [ ] Worker model family matches paper (14-layer U-Net score map).
- [ ] Default environment/resolution and evaluation metrics match paper settings.

## 7. Notes on Ambiguities To Resolve During Implementation

The PDF text extraction is complete enough for protocol-level reproduction, but some table cells and fine equation typography are image-heavy. If strict numeric parity is required, verify directly from page figures/tables in `Paper.pdf` before final benchmark claims.
