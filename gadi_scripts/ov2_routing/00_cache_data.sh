#!/bin/bash
# Cache CALVIN-1p and DROID-1p datasets on a GADI LOGIN node (which has
# internet). Compute nodes are offline, so this MUST be run first, before any
# qsub training/eval job. Run directly (NOT via qsub):
#
#   bash gadi_scripts/ov2_routing/00_cache_data.sh
#
set -euo pipefail

module purge
module load pytorch/2.12.0

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main
source .venv-ov2/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ka69/yc0686/hf_cache
mkdir -p "$HF_HOME"

# Online on the login node so the datasets download into HF_HOME.
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

python - <<'PY'
from datasets import load_dataset

print(">> CALVIN 1p (train + validation)")
load_dataset("ACIDE/AHA-Calvin-1p", split="train")
load_dataset("ACIDE/AHA-Calvin-1p", split="validation")

print(">> DROID 1p (train_1k + bench)")
load_dataset("ACIDE/DROID_1p_1k", split="train")
load_dataset("ACIDE/DROID_1p_bench", split="train")

print("All CALVIN/DROID 1p splits cached into HF_HOME.")
PY

echo "Done. HF_HOME=$HF_HOME is now populated; compute-node jobs can run offline."
