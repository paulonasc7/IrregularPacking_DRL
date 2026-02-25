#!/bin/bash
# Setup script for Paperspace Gradient (Ubuntu, CUDA GPU machine)
# Run once after uploading the project:
#   bash setup_paperspace.sh

set -e

echo "=== Irregular-Object-Packing: Paperspace Setup ==="

# ── 1. Python venv ──────────────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip --quiet

# ── 2. Detect CUDA version and install matching PyTorch ─────────────────────
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "none")
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
echo "Detected CUDA: $CUDA_VER"

if [[ "$CUDA_MAJOR" == "12" ]]; then
    # Select the closest PyTorch wheel tag for the detected CUDA 12.x minor version.
    # Using the closest matching wheel avoids ABI mismatches between libcusparse and
    # libnvJitLink that cause "undefined symbol: __nvJitLinkAddData_12_1" on Paperspace.
    if   [[ "$CUDA_MINOR" -ge 6 ]]; then WHL_TAG="cu126"
    elif [[ "$CUDA_MINOR" -ge 4 ]]; then WHL_TAG="cu124"
    else                                  WHL_TAG="cu121"
    fi
    echo "Installing PyTorch with wheel tag: $WHL_TAG"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$WHL_TAG" --quiet
elif [[ "$CUDA_MAJOR" == "11" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "WARNING: Could not detect CUDA version. Installing CPU-only PyTorch."
    pip install torch torchvision --quiet
fi

# ── 3. Install remaining requirements (torch already installed above) ────────
# Install everything except torch/torchvision (already installed above)
grep -vE "^torch" requirements.txt | pip install -r /dev/stdin --quiet

# ── 4. Install pybullet-object-models in editable mode ──────────────────────
pip install -e ./pybullet-object-models-master --quiet

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run training with:"
echo "  source .venv/bin/activate"
echo "  python scripts/train_hrl_packing.py --num_workers 1 --replay_size 1000 --stage1_episodes 30 --eps_end 0.1 --log_every 5"
