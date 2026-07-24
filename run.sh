#!/bin/bash
set -eo pipefail
export HF_HOME=${HF_HOME:-/tmp/hf}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== env ==="
nvidia-smi || true
python3 -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"

pip install --quiet --no-cache-dir peft datasets accelerate 2>&1 | tail -1 || pip install --quiet peft datasets accelerate

python3 main.py
