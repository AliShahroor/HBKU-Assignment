#!/bin/bash
# Separate environment for the MS-SWIFT DPO experiment (keeps the main .venv untouched).
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv-swift
.venv-swift/bin/pip install --upgrade pip
.venv-swift/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-swift/bin/pip install "ms-swift" msgspec -c <(echo "torch==2.11.0")
.venv-swift/bin/pip freeze > report/pip_freeze_swift.txt
.venv-swift/bin/python -c "import swift,torch,transformers,trl,peft;print('OK',swift.__version__,torch.__version__,torch.cuda.is_available(),transformers.__version__,trl.__version__,peft.__version__)"
