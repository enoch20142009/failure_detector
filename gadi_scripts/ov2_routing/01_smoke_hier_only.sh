#!/bin/bash
#PBS -P ka69
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Hierarchical dual-query router smoke only (30 samples, 1 epoch).
#
#   qsub gadi_scripts/ov2_routing/01_smoke_hier_only.sh
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

echo "==== hierarchical routed + hybrid pooling smoke (CALVIN) ===="
python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --dataset_name calvin --pov 1 \
  --num_entry 30 --num_epochs 1 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --use_hier_router --router_mode contrastive \
  --vision_layer_indices 9 17 24 \
  --pooling_mode hybrid \
  --loss_mode fusion --prediction_mode fusion \
  --gate_style multiplicative \
  --result_folder ./results_ov2_smoke_hier_routed

echo "Smoke complete."
