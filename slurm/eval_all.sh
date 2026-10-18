#!/bin/bash
# Submits eval jobs for all 8 benchmarks with the leaderboard settings (t=0.6, top_p=0.95, 8 samples, 32K tokens).
# Big benchmarks are split into chunks of CHUNK questions (default 60) so every job fits in its time limit;
# scripts/summarize.py merges the chunks back together afterwards.
#
# Run from the repo root with plain bash, not sbatch (this script only submits jobs):
#   TIME=2:00:00 bash slurm/eval_all.sh Qwen/Qwen2.5-3B-Instruct base   (the base model's answers are short)
#   bash slurm/eval_all.sh /scratch/$USER/p1/models/concise5_fal advanced_v3
#
# Optional settings:
#   PARTITION=a30_normal_q     use A30s instead of L40S (on TinkerCliffs: h200_normal_q)
#   AFTER=<job id>             queue now, start once that training job finishes successfully
#   AFTER=afterany:<job id>    start once it ends for any reason, e.g. it hit the time limit
#   ONLY="math minerva"        only run these benchmarks
#
# Results go to /scratch/$USER/p1/eval/<name>. Benchmarks and chunks that already finished are skipped,
# so you can just run this again after a timeout. Then:
#   python scripts/summarize.py <name>=/scratch/$USER/p1/eval/<name> ...
set -eo pipefail

MODEL=$1
NAME=$2
PARTITION=${PARTITION:-l40s_normal_q}
case $AFTER in *:*) DEP=$AFTER ;; *) DEP=afterok:$AFTER ;; esac
CLUSTER=tc; [[ $(hostname) == falcon* ]] && CLUSTER=fal   # QOS names look like fal_l40s_normal_short
[ -n "$NAME" ] || { echo "usage: bash slurm/eval_all.sh <model path or HF id> <name>"; exit 1; }
OUT=/scratch/$USER/p1/eval/$NAME
mkdir -p "$OUT"
cd "$(dirname "$0")/../eval"   # eval_single.sh expects to find data/ and prompts/ here

CHUNK=${CHUNK:-60}
for d in ${ONLY:-$(ls data)}; do
  N=$(grep -c . "data/$d/test.jsonl")
  if find "$OUT" -path "*/$d/*_s0_e${N}.jsonl" | grep -q .; then echo "skip $d: already evaluated"; continue; fi
  for ((s = 0; s < N; s += CHUNK)); do
    e=$(( s + CHUNK < N ? s + CHUNK : N ))
    # The short QOS gets higher priority than the default one, with a 1 day max
    MODEL=$MODEL OUTPUT_DIR=$OUT CONDA_ENV=p1eval START_IDX=$s END_IDX=$e \
      sbatch --job-name="ev-$NAME-$d-$s" --time="${TIME:-4:00:00}" --partition="$PARTITION" \
        --qos="${CLUSTER}_${PARTITION%_q}_short" ${AFTER:+--dependency=$DEP} --output="$OUT/slurm-$d-$s-%j.log" eval_single.sh "$d"
  done
done
echo "Done submitting. Watch: squeue -u \$USER"
