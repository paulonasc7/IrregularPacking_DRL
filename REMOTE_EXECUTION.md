# Remote Execution on Paperspace Gradient

How to drive the Paperspace GPU notebook from your local machine using `remote.py`. The script talks to the notebook's Jupyter contents and kernels APIs over HTTPS + websocket, so any local terminal that can reach the notebook URL works.

---

## Infrastructure

| Item | Value |
|------|-------|
| Paperspace URL | Hardcoded in `remote.py` (changes when the notebook restarts) |
| Token | Hardcoded in `remote.py` (env var `PAPERSPACE_TOKEN` overrides) |
| Remote project path | `/notebooks/IrregularPacking_DRL/` |
| GPU | NVIDIA RTX A4000, 16 GB VRAM |

`remote.py` is `.gitignored` (it stores the URL and token in plaintext). It only ever lives on your local machine. Each new notebook session needs the URL and token updated.

Local dependencies (only needed to run `remote.py`, not the training code):

```bash
pip install requests aiohttp
```

---

## Command Reference

```bash
# List remote files
python remote.py ls [remote_path]

# Upload a single file
python remote.py upload <local_path> [remote_path]

# Sync all relevant project files (see filter below)
python remote.py sync . IrregularPacking_DRL

# Sync only Python sources (faster iterate-on-code loop)
python remote.py sync . IrregularPacking_DRL --ext .py

# Run a shell command on the notebook (streams output line-by-line)
python remote.py run "<command>" --cwd /notebooks/IrregularPacking_DRL [--timeout 1800]

# Execute arbitrary Python in a fresh remote kernel
python remote.py exec "<python code>" [--timeout 1800]

# Download a result file
python remote.py download IrregularPacking_DRL/<file> [local_path]

# Kernel management
python remote.py kernels          # list running kernels
python remote.py kill <kernel_id> # stop a kernel
```

### What `sync` uploads and skips

`cmd_sync` walks the local tree and uploads everything that survives the following filters:

- `SKIP_DIRS` — directory names skipped wherever they appear: `.git`, `__pycache__`, `.venv`, `venv`, `env`, `.ipynb_checkpoints`, `.mypy_cache`, `results`, **`trash`**, **`pybullet-object-models-master`**.
- `SKIP_REL_PATHS` — project-relative subtrees: `logs/data`, `logs/models`, `logs/visualizations`, `logs/transitions`.
- `SKIP_SUFFIXES` — file extensions: `.pt`, `.ckpt`, `.npy`, `.pstats`, `.pdf`.
- `SKIP_FILENAMES` — `.DS_Store`, `Thumbs.db`, `remote.py` itself.
- `--ext` — optional whitelist of suffixes (passes only those extensions).

The YCB mesh tree (`pybullet-object-models-master/`) is intentionally **not** synced — it's ~50 MB of small binary files, and the Jupyter contents API uploads one file per HTTP PUT. Clone it directly on Paperspace (see one-time setup below).

---

## One-time setup on a fresh notebook

```bash
# 1. From local: push code
python remote.py sync . IrregularPacking_DRL

# 2. On the remote (run via `remote.py run` or in the Jupyter terminal):
#    Clone the YCB mesh repo into the project so the training scripts find it.
python remote.py run "git clone https://github.com/eleramp/pybullet-object-models.git pybullet-object-models-master" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 300

# 3. Install dependencies (CUDA-matched PyTorch wheel + requirements.txt).
python remote.py run "bash setup_paperspace.sh" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 900

# 4. Sanity-check the GPU.
python remote.py run "nvidia-smi"
```

`setup_paperspace.sh` creates a `.venv/`. Subsequent `run` commands need to activate it explicitly (e.g. `source .venv/bin/activate && python ...`) or just call `.venv/bin/python` directly.

---

## Project commands

This project's entry points are under `scripts/`. Manager + worker hierarchical training:

```bash
python scripts/train_hrl_packing.py \
    --obj_dir pybullet-object-models-master \
    --episodes 300 --stage1_episodes 100 \
    --num_objects 50 --resolution 200 \
    --box_size 0.256 0.256 0.256 \
    --save_path logs/packing_hrl.pt \
    --checkpoint_path logs/checkpoint_hrl.pt
```

Worker-only training (no manager):

```bash
python scripts/train_packing.py --obj_dir pybullet-object-models-master
```

Evaluation with a trained checkpoint:

```bash
python scripts/eval_packing.py \
    --obj_dir pybullet-object-models-master \
    --model_path logs/packing_hrl.pt \
    --episodes 5 --num_objects 10
```

Heuristic baseline (no model) for comparison:

```bash
python scripts/eval_packing.py --obj_dir pybullet-object-models-master --episodes 5
```

### Examples wrapped in `remote.py`

```bash
# Quick smoke test: 10 episodes, short pretrain, prints every episode
python remote.py run \
    ".venv/bin/python scripts/train_hrl_packing.py --episodes 10 --stage1_episodes 3 --num_objects 20 --log_every 1" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 1800

# Evaluate the bundled checkpoint on 5 episodes
python remote.py run \
    ".venv/bin/python scripts/eval_packing.py --model_path logs/packing_hrl.pt --episodes 5 --num_objects 10" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 600

# Check GPU
python remote.py run "nvidia-smi"
```

---

## Long-running training (detach + tail)

A websocket-bound `remote.py run` has a hard timeout. Full training runs (hundreds of episodes, multi-hour) should be **detached** on the remote and monitored by tailing a log file:

```bash
# 1. Launch training detached on the remote
python remote.py run \
    "nohup .venv/bin/python scripts/train_hrl_packing.py --episodes 300 --stage1_episodes 100 \
        > logs/run.log 2>&1 & echo started PID=\$!" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 60

# 2. Watch progress (run can be Ctrl-C'd locally; the remote keeps training)
python remote.py run "tail -f logs/run.log" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 3600

# 3. Check whether it's still running / find its PID
python remote.py run "pgrep -af train_hrl_packing" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 30

# 4. Stop it if you need to
python remote.py run "pkill -f train_hrl_packing" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 30

# 5. Pull the trained checkpoint back to local
python remote.py download IrregularPacking_DRL/logs/packing_hrl.pt logs/packing_hrl.pt
```

`train_hrl_packing.py` writes a resumable snapshot every `--checkpoint_every` episodes (default 50) to `--checkpoint_path` (default `logs/checkpoint_hrl.pt`). To pick up after the notebook reboots:

```bash
python remote.py run \
    ".venv/bin/python scripts/train_hrl_packing.py --resume logs/checkpoint_hrl.pt --episodes 300" \
    --cwd /notebooks/IrregularPacking_DRL --timeout 1800
```

---

## Iterate-on-code loop

```
1. Edit code locally
2. python remote.py sync . IrregularPacking_DRL --ext .py
3. python remote.py run \
       ".venv/bin/python scripts/train_hrl_packing.py --episodes 10 --stage1_episodes 3 --num_objects 20 --log_every 1" \
       --cwd /notebooks/IrregularPacking_DRL --timeout 1800
4. Read streamed stdout
5. python remote.py download IrregularPacking_DRL/logs/packing_hrl.pt logs/packing_hrl.pt
6. Reason about results → 1
```

---

## Output files

Produced by `scripts/train_hrl_packing.py` under `/notebooks/IrregularPacking_DRL/logs/` on the remote:

- `packing_hrl.pt` — best-moving-packed-count checkpoint (manager + worker state dicts + arch info)
- `checkpoint_hrl.pt` — full resumable snapshot (adds optimizer/scheduler/RNG state; included replay buffers if `--checkpoint_with_replay`)

Produced by `scripts/eval_packing.py`: stdout summary lines per episode (no files unless you redirect).

---

## Notes & gotchas

- **Streaming output.** `cmd_run` now launches the remote subprocess with `PYTHONUNBUFFERED=1` and `stderr` merged into `stdout`, so per-episode `print(...)` lines from training arrive locally as they're emitted. If a tool you call insists on block-buffering, prefix with `stdbuf -oL ` or use `python -u`.
- **Token / URL expiry.** The Paperspace notebook URL changes whenever the instance is restarted. Update the constants at the top of `remote.py` (or export `PAPERSPACE_URL` / `PAPERSPACE_TOKEN`) for each new session.
- **Default `--timeout` is 1800 s (30 min).** Anything longer should use the detach + tail pattern above; the websocket has its own deadline regardless of what you pass.
- **Kernel lifecycle.** Each `run` / `exec` starts a fresh kernel and deletes it when the call returns. There is no in-memory state between calls — environment variables, in-Python imports, etc. don't persist.
- **Paperspace sleep.** Idle notebooks pause. Wake them from the Paperspace dashboard before retrying.
- **YCB meshes are not synced.** They live under `pybullet-object-models-master/` and are excluded from `sync`. Clone them on the remote once (see one-time setup).
- **Local-only files.** `remote.py`, `.DS_Store`, `Thumbs.db`, `Paper.pdf`, and any `.pt` / `.ckpt` / `.npy` / `.pstats` files are excluded from `sync` by `SKIP_FILENAMES` / `SKIP_SUFFIXES`. The `trash/` folder and the regenerable `logs/{data,models,visualizations,transitions}/` subtrees are excluded by `SKIP_DIRS` / `SKIP_REL_PATHS`.
