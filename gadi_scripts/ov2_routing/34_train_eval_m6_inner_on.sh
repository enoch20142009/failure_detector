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
# M6 inner-on — contrastive outer + per-patch inner guiding (revert toward E).
#   qsub gadi_scripts/ov2_routing/34_train_eval_m6_inner_on.sh
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
RESULT_DIR=./results_ov2_calvin_m6_cod_alpha_inner_on
EVAL_IN=./eval_results/ov2_calvin_m6_cod_alpha_inner_on_indomain
EVAL_X=./eval_results/ov2_calvin_m6_cod_alpha_inner_on_to_droid

python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" --dataset_name calvin --pov 1 \
  --num_epochs 5 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --use_nested_guided_fusion \
  --ngf_layer_weight_mode contrastive \
  --vision_layer_indices 6 12 18 \
  --ngf_tap block --use_merger_adapter --merger_adapter_rank 64 \
  --router_mode contrastive --gate_type sigmoid --gate_style guiding \
  --layer_balance_coef 0.01 \
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1 \
  --result_folder "${RESULT_DIR}"

python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" --fs_id "${RESULT_DIR}" \
  --dataset_name calvin --pov 1 --split test --batch_size 1 \
  --prediction_mode fusion --grounding_check --result_folder "${EVAL_IN}"

python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" --fs_id "${RESULT_DIR}" \
  --dataset_name droid --pov 1 --split test --batch_size 1 \
  --prediction_mode fusion --grounding_check --result_folder "${EVAL_X}"

echo "Done M6 inner-on: ${RESULT_DIR}"
