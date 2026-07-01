#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Baseline MaTCA head on DROID-1p: train in-domain, then evaluate in-domain
# (DROID) and cross-dataset (CALVIN).
#
#   qsub gadi_scripts/ov2_routing/04_train_eval_droid_baseline.sh
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

RESULT_DIR=./results_ov2_droid_baseline
EVAL_IN=./eval_results/ov2_droid_baseline_indomain
EVAL_X=./eval_results/ov2_droid_baseline_to_calvin

echo "==== train baseline on DROID-1p ===="
python src/finetune_FS_ov2_routed.py \
  --dataset_name droid --pov 1 \
  --num_epochs 5 --batch_size 2 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --vlm_model_id /scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1 \
  --result_folder "${RESULT_DIR}"

echo "==== eval in-domain (DROID-1p) ===="
python src/evaluate_FS_ov2_routed.py \
  --fs_id "${RESULT_DIR}" --dataset_name droid --pov 1 --split test \
  --vlm_model_id /scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct \
  --batch_size 1 --prediction_mode fusion --result_folder "${EVAL_IN}"

echo "==== eval cross-dataset (CALVIN-1p) ===="
python src/evaluate_FS_ov2_routed.py \
  --fs_id "${RESULT_DIR}" --dataset_name calvin --pov 1 --split test \
  --vlm_model_id /scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct \
  --batch_size 1 --prediction_mode fusion --result_folder "${EVAL_X}"

echo "Done: ${RESULT_DIR} ${EVAL_IN} ${EVAL_X}"
