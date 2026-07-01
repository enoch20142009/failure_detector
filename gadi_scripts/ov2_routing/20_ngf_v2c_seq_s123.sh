#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=16:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# v2: inner residual h_l = V_l + beta_l * delta_l
# Arch C multi-seed (seed 123). Sequential depth-recurrent refinement, identical
# to 19_ngf_v2c_sequential.sh but seed=123, to estimate run-to-run variance
# (combine with the seed=42 run in results_ov2_calvin_ngf_v2c_seq).
#
#   qsub gadi_scripts/ov2_routing/20_ngf_v2c_seq_s123.sh
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
SEED=123

RESULT_DIR=./results_ov2_calvin_ngf_v2c_seq_s${SEED}
EVAL_IN=./eval_results/ov2_calvin_ngf_v2c_seq_s${SEED}_indomain
EVAL_X=./eval_results/ov2_calvin_ngf_v2c_seq_s${SEED}_to_droid

echo "==== Arch C seed ${SEED}: sequential depth-recurrent refinement on CALVIN-1p ===="
python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --dataset_name calvin --pov 1 --seed ${SEED} \
  --num_epochs 5 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --use_nested_guided_fusion --ngf_sequential \
  --vision_layer_indices 9 17 24 \
  --router_mode contrastive --gate_type sigmoid --gate_style guiding \
  --use_merger_adapter --merger_adapter_rank 64 \
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1 \
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

echo "Done: ${RESULT_DIR} ${EVAL_IN} ${EVAL_X}"
