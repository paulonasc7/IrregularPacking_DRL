# HRL Refactor Plan (Robot-Agnostic Packing)

## Objective
Refactor this repository from a push/grasp robot loop into a pure irregular-object packing planner that follows the paper design:
- Manager policy: choose the next object.
- Worker policy: choose placement pose for that object.
- Physics-backed reward and feasibility checks (no robot execution dependency).

## Scope
- Keep: PyBullet physics, object models, heightmap representation idea.
- Remove from main execution path: robot control abstraction, push/grasp action loop, camera-dependent manipulation logic.
- Build: a clean hierarchical RL training/evaluation pipeline for packing only.

## Target Architecture

### 1. `packing/env_packing.py`
Primary environment for packing episodes.

Key responsibilities:
- Load object set for an episode.
- Maintain packed/unpacked sets.
- Maintain container state (`box_heightmap` + occupancy/stability info).
- Apply placement action and return transition.

Proposed API:
```python
class PackingEnv:
    def reset(self, object_ids: list[int] | None = None) -> dict: ...
    def get_state(self) -> dict: ...
    def candidate_orientations(self, item_id: int) -> np.ndarray: ...
    def step(self, item_id: int, placement: np.ndarray) -> tuple[dict, float, bool, dict]: ...
```

### 2. `packing/state.py`
State builders for manager/worker.

Proposed API:
```python
def build_manager_state(env_state: dict) -> dict: ...
def build_worker_state(env_state: dict, item_id: int) -> dict: ...
```

Expected tensors:
- Manager: container heightmap + remaining object descriptors.
- Worker: container heightmap + selected object oriented heightmap channels (`Ht/Hb`) + mask.

### 3. `packing/reward.py`
Paper-aligned reward composition.

Proposed API:
```python
def compactness(env_state: dict) -> float: ...
def pyramidality(env_state: dict) -> float: ...
def stability(success: bool, sim_info: dict) -> float: ...
def total_reward(env_state: dict, success: bool, sim_info: dict, weights: dict) -> float: ...
```

### 4. `packing/models_manager.py`
Manager network: object-selection Q/policy head.

Proposed API:
```python
class SequenceQNet(nn.Module):
    def forward(self, container_feat, object_feats, valid_mask): ...
```

Output:
- Per-object Q-values or logits over remaining objects.

### 5. `packing/models_worker.py`
Worker network: placement prediction for selected object.

Proposed API:
```python
class PlacementQNet(nn.Module):
    def forward(self, worker_state_tensor): ...
```

Output:
- Q-map over `(orientation_idx, y, x)` or direct parameterized pose head.

### 6. `packing/agent_hrl.py`
Orchestrates hierarchical decision + training target generation.

Proposed API:
```python
class HRLPackingAgent:
    def act_manager(self, manager_state, epsilon: float): ...
    def act_worker(self, worker_state, epsilon: float): ...
    def remember(self, transition: dict): ...
    def train_step(self): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
```

### 7. `packing/replay.py`
Replay buffers (can be separate per hierarchy level).

Proposed API:
```python
class ReplayBuffer: ...
class HierarchicalReplay:
    manager: ReplayBuffer
    worker: ReplayBuffer
```

### 8. `scripts/train_packing.py`
Entry point for training.

Responsibilities:
- Config load.
- Episode loop.
- Manager/worker action selection.
- Environment transitions.
- Logging + checkpointing.

### 9. `scripts/eval_packing.py`
Inference-only evaluation.

Metrics:
- Pack success rate.
- Mean packed volume ratio.
- Mean reward.
- Mean steps to completion.
- Runtime per episode.

## Migration Map (Current -> New)

- `main.py`:
  - Replace with thin launcher that calls `scripts/train_packing.py` or deprecate.
- `trainer.py`:
  - Split into manager/worker trainers in `packing/agent_hrl.py`.
- `models.py`:
  - Split into `models_manager.py` + `models_worker.py`.
- `env.py`:
  - Keep reusable geometry/heightmap routines; wrap/refactor into `env_packing.py`.
- `heuristics_HM.py`:
  - Keep only as optional baseline comparator (not main training dependency).
- `robot.py`, `logger.py`:
  - Remove from training critical path; keep optional tooling only.

## Implementation Phases

### Phase 1: Environment foundation
- Implement `PackingEnv.reset/get_state/step`.
- Ensure deterministic episode setup via seed.
- Add container/object validity checks.

### Phase 2: State + reward
- Implement manager/worker state encoders.
- Implement compactness/pyramidality/stability rewards.
- Unit-check reward monotonicity on synthetic cases.

### Phase 3: Models
- Implement manager and worker Q networks.
- Define action masks for valid objects and valid placement cells.

### Phase 4: HRL training loop
- Build hierarchical rollout and replay storage.
- TD targets for both levels.
- Epsilon schedules for manager and worker separately.

### Phase 5: Evaluation + baselines
- Inference-only script.
- Compare against heuristic baseline (`heuristics_HM`-style placement).

### Phase 6: Cleanup
- Deprecate robot-centric path from README.
- Add dependency file and reproducible run commands.

## Minimal File Tree (Target)
```text
packing/
  __init__.py
  env_packing.py
  state.py
  reward.py
  replay.py
  models_manager.py
  models_worker.py
  agent_hrl.py
scripts/
  train_packing.py
  eval_packing.py
configs/
  packing_default.yaml
tests/
  test_env_packing.py
  test_reward.py
```

## Acceptance Criteria
- Can train for `N` episodes without robot module imports.
- Can evaluate a fixed set of 5 objects in inference-only mode.
- Produces stable metrics across multiple seeds.
- Runtime is dominated by worker inference + physics checks, not logging/manipulation overhead.

## First Practical Milestone
Implement an inference-only prototype:
1. Load a fixed 5-object set.
2. Use heuristic manager (largest-first) + learned/heuristic worker.
3. Run until done or step cap.
4. Report success rate and episode time.

This gives immediate value while full HRL training matures.
