#!/bin/bash
#PBS -P ka69
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=50GB
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Phase 0: vision-only per-(layer,module) linear probe of the OV2 ViT.
#   Frozen ViT only (no LM forward, no backprop). Taps Act / LN2 / RC2 per
#   layer, mean-pools tokens, and reports two VALID diagnostics:
#     1. CALVIN in-domain FAILURE AUROC (success vs fail; only real 2-class set).
#     2. DOMAIN-invariance: CALVIN-vs-DROID and CALVIN-vs-AHA separability
#        (lower = more shift-invariant = the paper's claimed Act/intermediate
#        strength).
#   (A CALVIN->DROID *failure* probe is degenerate for vision: DROID has no real
#    visual failures; the eval augmentation only changes task TEXT.)
#
#   qsub gadi_scripts/ov2_routing/24_probe_vit_modules_phase0.sh
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
RESULT_DIR=./eval_results/ov2_probe_phase0

echo "==== Phase 0 probe: CALVIN ID failure + CALVIN-vs-DROID/AHA domain invariance ===="
python src/probe_vit_modules.py \
  --vlm_model_id "${VLM_MODEL}" \
  --pov 1 --style image \
  --num_entry 800 \
  --max_pixels 200704 \
  --result_folder "${RESULT_DIR}"

echo "Done: ${RESULT_DIR}"
