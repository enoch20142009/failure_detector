#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Grounding / shortcut audit on the (already-trained) head-only baselines.
#
# The baseline train/eval jobs (02, 04) ran WITHOUT --grounding_check, so we have
# no measure of how much the strong in-domain baselines rely on the INSTRUCTION
# vs visual SHORTCUTS. This eval-only job re-runs both baseline checkpoints with
# the query-swap grounding probe on in-domain AND cross-dataset splits.
#
#   grounding_flip_rate high  -> prediction depends on the instruction (grounded)
#   grounding_flip_rate ~0    -> instruction ignored; model uses a visual shortcut
#
# Compare these flip rates against the routed/MoE runs (03/05/06/07) to test
# whether grounded routing increases instruction-sensitivity. Motivated by
# VQA_TECHNIQUES_AND_RECOMMENDATIONS.md (hard negatives / grounding / shortcuts).
#
#   qsub gadi_scripts/ov2_routing/08_grounding_audit_baselines.sh
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

CALVIN_BASE=./results_ov2_calvin_baseline
DROID_BASE=./results_ov2_droid_baseline

echo "==== [CALVIN baseline] in-domain CALVIN + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${CALVIN_BASE}" --dataset_name calvin --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder ./eval_results/ov2_calvin_baseline_indomain_grounding

echo "==== [CALVIN baseline] cross DROID (balanced) + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${CALVIN_BASE}" --dataset_name droid --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder ./eval_results/ov2_calvin_baseline_to_droid_grounding

echo "==== [DROID baseline] in-domain DROID (balanced) + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${DROID_BASE}" --dataset_name droid --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder ./eval_results/ov2_droid_baseline_indomain_grounding

echo "==== [DROID baseline] cross CALVIN + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${DROID_BASE}" --dataset_name calvin --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder ./eval_results/ov2_droid_baseline_to_calvin_grounding

echo "Done: grounding audit complete."
