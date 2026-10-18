# TML-Group1: Reasoning SFT (ECE 6514 Project 1)

Michael Arthur, Kiara Klevar, Joseph Leonard, Gia Subedi

We fine-tune `Qwen/Qwen2.5-3B-Instruct` (full-parameter SFT) on 15K samples picked from the 100K AceReason-1.1-SFT subset
`redsgnaoh/acereason11_100k`, and try to beat a model trained on 15K random samples.
Assignment: [reds-lab/Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM).
The full write-up is in [report/report.md](report/report.md).

## Results

Our submitted model is **Advanced-v3**: math-only samples with exactly one final `\boxed{}` answer, no proofs, no truncation,
and solutions between 1K and 6K tokens, trained for 5 epochs.

| Model | Pass@1 avg | Pass@8 avg | No answer |
|---|---:|---:|---:|
| Base (Qwen2.5-3B-Instruct) | 27.6 | 52.9 | 2.6% |
| Random | 27.2 | 54.7 | 19.4% |
| Advanced-v1 | 28.5 | 58.2 | 18.1% |
| Advanced-v2 | 28.9 | 57.7 | 13.1% |
| **Advanced-v3** (submitted) | **31.6** | **59.6** | 3.6% |

Averages over AIME24, AMC, CN-Math-24, GPQA, Kaoyan, MATH-500, Minerva and OlympiadBench, with 8 samples per question
at temperature 0.6, top-p 0.95 and up to 32K tokens. Per-benchmark tables are in the report.

![Average Pass@1](report/pass1.png)
![Average Pass@8](report/pass8.png)
![Average no-answer rate](report/no_answer.png)

| | Hugging Face |
|---|---|
| Submitted model | [josephl04/ece6514-p1-advanced-v3](https://huggingface.co/josephl04/ece6514-p1-advanced-v3) |
| Random baseline model | [josephl04/ece6514-p1-random](https://huggingface.co/josephl04/ece6514-p1-random) |
| Random dataset | [josephl04/ece6514-p1-random-15k](https://huggingface.co/datasets/josephl04/ece6514-p1-random-15k) |
| Advanced dataset (used for Advanced-v2 and Advanced-v3) | [josephl04/ece6514-p1-concise-15k](https://huggingface.co/datasets/josephl04/ece6514-p1-concise-15k) |
| Advanced-v1 dataset | [josephl04/ece6514-p1-advanced-15k](https://huggingface.co/datasets/josephl04/ece6514-p1-advanced-15k) |

## What's in the repo

| Path | What it does |
|---|---|
| `scripts/select_data.py` | Builds the 15K datasets (`random`, `advanced`, `concise`) and can push them to Hugging Face |
| `configs/train.yaml`, `configs/ds_z3_config.json` | LLaMA-Factory training config (DeepSpeed ZeRO-3) |
| `data/dataset_info.json` | Tells LLaMA-Factory how to read our datasets, including the `system` column |
| `slurm/train.sh` | Slurm job for training |
| `slurm/eval_all.sh` | Submits eval jobs for all 8 benchmarks, split into chunks |
| `eval/` | The course eval harness, with a few memory fixes (see below) |
| `scripts/summarize.py` | Turns eval outputs into the pass@1 / pass@8 / no-answer tables |
| `scripts/upload_model.py` | Uploads a trained model to Hugging Face with a model card |
| `setup/arc_setup.sh` | Builds the two conda environments on ARC |
| `tests/` | Unit tests for data selection and scoring: `python -m pytest tests` (no GPU needed) |
| `report/` | The report |

## How to reproduce

Everything runs on VT ARC through Slurm, from the repo root. We used the Falcon cluster (`falcon1.arc.vt.edu`)
with 2 L40S GPUs for training and L40S/A30 GPUs for evaluation.

**1. Set up the environments (once, about 30 minutes).** This creates `p1train` for training and `p1eval` for evaluation,
and asks for your Hugging Face token (write access) and W&B key at the end.
```bash
bash setup/arc_setup.sh
```

**2. Build the datasets.** Each command writes `data/<strategy>_15k.json` and uploads it to Hugging Face.
```bash
module load Miniconda3 && source activate p1train
HF_USER=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")
python scripts/select_data.py --strategy random   --push $HF_USER/ece6514-p1-random-15k     # Random
python scripts/select_data.py --strategy advanced --push $HF_USER/ece6514-p1-advanced-15k   # Advanced-v1
python scripts/select_data.py --strategy concise  --push $HF_USER/ece6514-p1-concise-15k    # Advanced-v2 and Advanced-v3
```

**3. Train.** Models are saved to `/scratch/$USER/p1/models/<name>`.
```bash
sbatch slurm/train.sh random_15k random_fal                         # Random
sbatch slurm/train.sh advanced_15k advanced_fal                     # Advanced-v1
sbatch slurm/train.sh concise_15k concise_fal                       # Advanced-v2
sbatch slurm/train.sh concise_15k concise5_fal num_train_epochs=5   # Advanced-v3
```
Random and Advanced-v1 need about 27-29 hours for 3 epochs, which is over ARC's 24-hour limit. We trained epoch 3 as a
second job from the epoch-2 checkpoint; the exact command is in the comments at the top of `slurm/train.sh`.

**4. Evaluate.** Run with plain `bash` (it submits one job per chunk). Finished chunks are skipped, so if a job times out,
just run the same command again.
```bash
TIME=2:00:00 bash slurm/eval_all.sh Qwen/Qwen2.5-3B-Instruct base
bash slurm/eval_all.sh /scratch/$USER/p1/models/concise5_fal advanced_v3
```

**5. Make the score tables.**
```bash
python scripts/summarize.py Base=/scratch/$USER/p1/eval/base Advanced-v3=/scratch/$USER/p1/eval/advanced_v3
```

**6. Upload a model.** The last argument is the number of epochs, which only goes into the model card.
```bash
python scripts/upload_model.py /scratch/$USER/p1/models/concise5_fal $HF_USER/ece6514-p1-advanced-v3 $HF_USER/ece6514-p1-concise-15k 5
```

## Changes from the assignment setup

- **Two conda environments.** The vLLM version the eval code needs and LLaMA-Factory want different `transformers` versions,
  so training and evaluation each get their own env.
- **Eval requirements.** `eval/requirements.txt` pins a `latex2sympy2` that doesn't import next to the required antlr4, so we
  install the copy bundled with Qwen2.5-Math, which is what the grader was written for. We also pin `datasets==3.6.0` and
  `pyarrow==19.0.1`, since newer pyarrow needs NumPy 2 and vLLM 0.6.1 needs NumPy < 2.
- **Eval memory fixes.** On 24-48 GB GPUs vLLM 0.6.1 crashed on long generations. `eval/eval.py` now draws the 8 samples
  as 8 single-sample rounds (same distribution), caps GPU memory at 88% on GPUs under 50 GB, and loads
  `chat_template.jinja` when the tokenizer config doesn't include the template. Prompts, sampling settings and grading
  are unchanged.
- **Training memory.** We don't use CPU offload or flash-attn. Instead we use PyTorch SDPA attention and Liger kernels,
  which compute the loss without building the full logits tensor, so 16K-token samples fit on a 48 GB L40S.
- **System prompt.** The assignment's `dataset_info.json` drops the dataset's `system` column, so training would use Qwen's
  default system prompt while the eval uses "Please reason step by step...". Ours maps the `system` column.
