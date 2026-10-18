"""Upload a trained model to Hugging Face in a form older transformers versions can still load.

Usage:
  python scripts/upload_model.py <model dir> <hf model repo> <hf dataset repo> [epochs]
  e.g.
  python scripts/upload_model.py /scratch/$USER/p1/models/concise5_fal <hf_user>/ece6514-p1-advanced-v3 \\
      <hf_user>/ece6514-p1-concise-15k 5

The optional epochs argument only goes into the model card (default 3).

Why this exists: our training environment (transformers 4.55) saves the chat template in a separate
chat_template.jinja file, but the course eval environment (transformers 4.45) only looks for it in
tokenizer_config.json. Without copying it over, apply_chat_template fails when the model is evaluated.
"""
import json
import sys
from pathlib import Path

CARD = """---
base_model: Qwen/Qwen2.5-3B-Instruct
datasets:
- {dataset}
library_name: transformers
---
# {repo}

Qwen2.5-3B-Instruct, full-parameter SFT on [{dataset}](https://huggingface.co/datasets/{dataset}), a 15K-sample
subset of redsgnaoh/acereason11_100k, for ECE 6514 Project 1 (TML Group 1). Trained with LLaMA-Factory for
{epochs} epochs, lr 5e-6 with a cosine schedule, global batch 8, cutoff_len 16384.
Code, data selection and evaluation: https://github.com/kiarak425/TML-Group1

Evaluate with the system prompt "Please reason step by step, and put your final answer within \\boxed{{}}."
"""


def embed_chat_template(model_dir):
    cfg_path, jinja = Path(model_dir) / "tokenizer_config.json", Path(model_dir) / "chat_template.jinja"
    cfg = json.loads(cfg_path.read_text())
    if "chat_template" not in cfg and jinja.exists():
        cfg["chat_template"] = jinja.read_text()
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    model_dir, repo, dataset, epochs = (sys.argv[1:] + ["3"])[:4]
    from huggingface_hub import HfApi
    embed_chat_template(model_dir)
    (Path(model_dir) / "README.md").write_text(CARD.format(repo=repo, dataset=dataset, epochs=epochs))
    api = HfApi()
    api.create_repo(repo, exist_ok=True)  # public, same as our datasets
    api.upload_folder(repo_id=repo, folder_path=model_dir, ignore_patterns=["checkpoint-*"])
    print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
