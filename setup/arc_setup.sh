#!/bin/bash
# One-time setup on ARC. Run it from the repo root on a login node:
#   bash setup/arc_setup.sh
# It takes about 20-30 minutes, mostly downloads. Running it again is fine; existing envs and logins are skipped.
#
# We use two conda envs because the vLLM version the eval code needs wants a different
# transformers version than LLaMA-Factory:
#   p1train  LLaMA-Factory (the copy pinned in the TA repo) + DeepSpeed, for training
#   p1eval   vLLM 0.6.1 + the math grader from eval/requirements.txt, for evaluation
set -eo pipefail

[ -f eval/requirements.txt ] || { echo "Run this from the repo root: bash setup/arc_setup.sh"; exit 1; }

module load Miniconda3
module load CUDA/12.6.0

# The LLaMA-Factory copy in the TA repo (it pins transformers<=4.55, deepspeed<=0.16.9, ...).
LLAMAFACTORY="git+https://github.com/reds-lab/Project-Reasoning-SFT-LLM.git@638989b9f1a02322fea5cf5b2be5042dbd99c209#subdirectory=LLaMA-Factory"
# eval/requirements.txt asks for latex2sympy2 1.9.1 from PyPI, which fails to import with the antlr4 version
# sympy needs. The grader was written for the copy that ships with Qwen2.5-Math, so we install that one.
LATEX2SYMPY="git+https://github.com/QwenLM/Qwen2.5-Math.git@a45202bd16f1ec06f433442dc1152d0074773465#subdirectory=evaluation/latex2sympy"

has_env() { conda env list | grep "^$1 " >/dev/null; }
# Use conda-forge, since the default Anaconda channel now asks you to accept its terms interactively.
new_env() { conda create -y -n "$1" -c conda-forge --override-channels "python=$2"; }

if has_env p1train; then
  echo "== p1train exists, skipping"
else
  echo "== Building p1train"
  new_env p1train 3.10
  source activate p1train
  pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
  # liger-kernel 0.6.2 is the version that works with transformers 4.55. --no-deps so it doesn't touch torch.
  pip install "llamafactory[metrics] @ $LLAMAFACTORY" deepspeed==0.16.8 wandb
  pip install --no-deps liger-kernel==0.6.2
  conda deactivate
fi

if has_env p1eval; then
  echo "== p1eval exists, skipping"
else
  echo "== Building p1eval"
  new_env p1eval 3.11
  source activate p1eval
  # We skip flash_attn: vLLM has its own attention kernels and building flash_attn takes hours.
  # transformers is pinned to a version from vLLM 0.6.1's time; it can still read tokenizers saved by p1train.
  # datasets and pyarrow are pinned because newer pyarrow needs NumPy 2, and vLLM 0.6.1 needs NumPy < 2.
  pip install -r <(grep -v -E '^(flash_attn|latex2sympy2)' eval/requirements.txt) \
    transformers==4.45.2 datasets==3.6.0 pyarrow==19.0.1 "latex2sympy2 @ $LATEX2SYMPY"
  conda deactivate
fi

# Logins. The tokens are saved in your home directory, never in this repo.
source activate p1train
if python -c "from huggingface_hub import whoami; whoami()" &>/dev/null; then
  echo "== Hugging Face: already logged in"
else
  echo "== Hugging Face login: paste a token with WRITE access (huggingface.co/settings/tokens)"
  huggingface-cli login
fi
if grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
  echo "== W&B: already logged in"
else
  echo "== W&B login: paste your API key (wandb.ai/authorize)"
  wandb login
fi
conda deactivate

echo "== Setup done. Next: python scripts/select_data.py, then sbatch slurm/train.sh"
