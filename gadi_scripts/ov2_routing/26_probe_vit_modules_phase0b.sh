#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=50GB
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Phase 0b: text-guided per-(layer,module) linear probe (+ Phase 0 baseline).
#   dgxa100: 16 CPUs per GPU (ncpus=16). Memory raised to 128GB after job
#   172516476 OOM at 64GB (patch_store retained full ViT tensors).
#   Grounding probe now streams one sample at a time (no patch_store).
#
#   qsub gadi_scripts/ov2_routing/26_probe_vit_modules_phase0b.sh
#
set -euo pipefail

module purge
module load pytorch/2.12.0

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main
source .venv-ov2/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ka69/yc0686/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VLM_MODEL=/scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct
RESULT_DIR=./eval_results/ov2_probe_phase0b

echo "==== Phase 0/0b: vision-only + text-guided layer probes (dgxa100) ===="
python src/probe_vit_modules.py \
  --probe_mode both \
  --text_pool_mode concat \
  --vlm_model_id "${VLM_MODEL}" \
  --pov 1 --style image \
  --num_entry 800 \
  --max_pixels 200704 \
  --result_folder "${RESULT_DIR}"

echo "Done: ${RESULT_DIR}/probe_results.json"
