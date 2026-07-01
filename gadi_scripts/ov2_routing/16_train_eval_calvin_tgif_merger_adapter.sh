#!/bin/bash
#PBS -P ka69
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=16:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# TGIF-style complete multi-depth vision feature mixture + merger adapter (no router).
# Stage 1: V_out = V_base + alpha * transform(sum_l w_l * V_l) over layers 9, 17, 24, base.
# Stage 1.5: trainable parallel adapter on frozen patch merger.
# Stage 2: MaTCA with task-conditioned pooling (same as job 12 baseline pairing).
#
# Compare against:
#   02  MaTCA baseline only
#   12  flat router + merger adapter + tcond
#   15  fuse-then-route + adapter + guiding (fusion + router)
#
#   qsub gadi_scripts/ov2_routing/16_train_eval_calvin_tgif_merger_adapter.sh
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

RESULT_DIR=./results_ov2_calvin_tgif_merger_adapter
EVAL_IN=./eval_results/ov2_calvin_tgif_merger_adapter_indomain
EVAL_X=./eval_results/ov2_calvin_tgif_merger_adapter_to_droid

echo "==== train TGIF fusion + merger adapter on CALVIN-1p ===="
python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --dataset_name calvin --pov 1 \
  --num_epochs 5 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --use_tgif_fusion \
  --vision_layer_indices 9 17 24 --fusion_mode static \
  --use_merger_adapter --merger_adapter_rank 64 \
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1 \
  --result_folder "${RESULT_DIR}"

echo "==== eval in-domain (CALVIN-1p) ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name calvin --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion \
  --result_folder "${EVAL_IN}"

echo "==== eval cross-dataset (DROID-1p) ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name droid --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion \
  --result_folder "${EVAL_X}"

echo "Done: ${RESULT_DIR} ${EVAL_IN} ${EVAL_X}"
