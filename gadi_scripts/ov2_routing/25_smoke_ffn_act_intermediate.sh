#!/bin/bash
#PBS -P ka69
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=50GB
#PBS -l walltime=00:40:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Smoke test for the Phase 1 FFN-Act intermediate-only path (cheap):
#   1 epoch on 8 samples. Validates that the FFN-Act hooks fire, the per-layer
#   down-projections + full trainable connector wire up, the grad-flow assert
#   passes (ViT checkpointing disabled), and a checkpoint saves/loads. Run this
#   before the full ~300 SU job.
#
#   qsub gadi_scripts/ov2_routing/25_smoke_ffn_act_intermediate.sh
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
RESULT_DIR=./results_ov2_smoke_ffn_act_intermediate

echo "==== SMOKE: NGF FFN-Act intermediate-only + full connector (8 samples, 1 epoch) ===="
python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --dataset_name calvin --pov 1 \
  --num_entry 8 --num_epochs 1 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --use_nested_guided_fusion --ngf_layer_weight_mode text \
  --vision_layer_indices 6 12 18 \
  --ngf_tap ffn_act --ngf_intermediate_only --ngf_full_connector \
  --router_mode contrastive --gate_type sigmoid --gate_style guiding \
  --layer_balance_coef 0.01 \
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1 \
  --result_folder "${RESULT_DIR}"

echo "==== SMOKE eval (8 samples) ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name calvin --pov 1 --split test \
  --num_entry 8 --batch_size 1 --prediction_mode fusion \
  --result_folder "./eval_results/ov2_smoke_ffn_act_intermediate"

echo "SMOKE DONE"
