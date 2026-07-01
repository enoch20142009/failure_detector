#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=05:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# M3: TGIF-style text-conditioned ViT depth fusion → flat contrastive router
#     → merger adapter → tcond MaTCA (matches M0 champion eval pipeline).
#
# Layers {9,17,24} include final depth; residual alpha init 0 preserves V_base.
# Compare target: M0 champion 0.832 DROID AUROC; fuse-then-route static 0.811.
#
#   qsub gadi_scripts/ov2_routing/29_m3_text_fuse_then_route_overnight.sh
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

RESULT_DIR=./results_ov2_calvin_m3_text_fuse_route
EVAL_IN=./eval_results/ov2_calvin_m3_text_fuse_route_indomain
EVAL_X=./eval_results/ov2_calvin_m3_text_fuse_route_to_droid
SMOKE_DIR=./results_ov2_smoke_m3_text_fuse_route

COMMON=(
  --vlm_model_id "${VLM_MODEL}"
  --dataset_name calvin --pov 1
  --target_layer_indices 19 28 36 --num_classifiers 3
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion
  --use_fuse_then_route
  --vision_layer_indices 9 17 24
  --fusion_mode text
  --router_mode contrastive --gate_type sigmoid --gate_style multiplicative
  --use_merger_adapter --merger_adapter_rank 64
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1
  --batch_size 1
)

echo "==== M3 smoke: text fuse-then-route (30 samples, 1 epoch) ===="
python src/finetune_FS_ov2_routed.py \
  "${COMMON[@]}" \
  --num_entry 30 --num_epochs 1 \
  --result_folder "${SMOKE_DIR}"

echo "==== M3 train: text fuse-then-route on CALVIN-1p (5 epochs) ===="
python src/finetune_FS_ov2_routed.py \
  "${COMMON[@]}" \
  --num_epochs 5 \
  --result_folder "${RESULT_DIR}"

echo "==== eval in-domain (CALVIN-1p) + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name calvin --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder "${EVAL_IN}"

echo "==== eval cross-dataset (DROID-1p) ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name droid --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder "${EVAL_X}"

echo "Done M3: ${RESULT_DIR} ${EVAL_IN} ${EVAL_X}"
