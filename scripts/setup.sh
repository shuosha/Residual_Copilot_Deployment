#!/bin/bash
set -e

echo "=== Setting up residual_copilot_hardware ==="

# Create and activate venv
uv venv --python 3.9 --seed
source .venv/bin/activate

# Install main dependencies (includes PyTorch 2.1.0+cu121)
uv sync

# Packages with build/dependency conflicts (require torch already installed)
uv pip install --no-deps rl-games==1.6.1
uv pip install gym==0.23.1
uv pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
uv pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

echo ""
echo "=== Setup complete. Run 'source .venv/bin/activate' to activate. ==="
