#!/bin/bash
# Environment setup for DSEG660 LoRA exercise.
# The machine's NVIDIA driver supports CUDA <= 12.9, so torch is pinned to the CUDA 12.8 build.
set -e
cd "$(dirname "$0")/.."
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
echo "torch==2.11.0" > /tmp/dseg660_constraints.txt
.venv/bin/pip install unsloth -c /tmp/dseg660_constraints.txt
.venv/bin/pip uninstall -y xformers || true
.venv/bin/pip freeze > report/pip_freeze.txt
.venv/bin/python -c "import unsloth,torch,transformers,trl,peft;print('OK',torch.__version__,torch.cuda.is_available(),transformers.__version__,trl.__version__,peft.__version__,unsloth.__version__)"
