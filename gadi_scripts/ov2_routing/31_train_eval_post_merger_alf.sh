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
# Post-merger ALF fusion (fuse-after-merger): each intermediate ViT depth goes
# through the frozen patch merger separately, then ALF-style cross-attention in
# 4096-d LLM space with the V_base merger output as the anchor. Compares against
# the M0 champion (flat dual-query router + merger adapter, CALVIN->DROID 0.832)
# and Ablation E (block-tap NGF, ~0.815).
#
# Intermediate depths come from 30_probe_layer_select.sh
# (eval_results/ov2_probe_layer_select/layer_select.json). Override with e.g.
#   VISION_LAYERS="6 12 18" qsub gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh
#
#   qsub gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh
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

VLM_MODEL="${VLM_MODEL:-/scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct}"
LAYER_SELECT_JSON="./eval_results/ov2_probe_layer_select/layer_select.json"

# Resolve intermediate depths: env override > layer_select.json > fallback {6,12,18}.
if [[ -n "${VISION_LAYERS:-}" ]]; then
  LAYERS="${VISION_LAYERS}"
  echo "Using VISION_LAYERS override: ${LAYERS}"
elif [[ -f "${LAYER_SELECT_JSON}" ]]; then
  LAYERS=$(python -c "import json; d=json.load(open('${LAYER_SELECT_JSON}')); print(' '.join(str(i) for i in d['recommended_intermediate_indices']))")
  echo "Using recommended layers from ${LAYER_SELECT_JSON}: ${LAYERS}"
else
  LAYERS="6 12 18"
  echo "layer_select.json not found; falling back to default layers: ${LAYERS}"
fi

TAG="post_merger_alf"
RESULT_DIR="./results_ov2_calvin_${TAG}"
EVAL_IN="./eval_results/ov2_calvin_${TAG}_indomain"
EVAL_X="./eval_results/ov2_calvin_${TAG}_to_droid"

echo "==== Post-merger ALF: train (CALVIN-1p), intermediates {${LAYERS}} + base(24) ===="
echo "RESULT_DIR=${RESULT_DIR}"

python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --dataset_name calvin --pov 1 \
  --num_epochs 5 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --use_post_merger_alf \
  --vision_layer_indices ${LAYERS} \
  --post_merger_adapter_rank 64 --alf_router_dim 256 \
  --layer_balance_coef "${LAYER_BALANCE_COEF:-0.01}" \
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1 \
  --result_folder "${RESULT_DIR}"

echo "==== eval in-domain (CALVIN-1p) + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name calvin --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder "${EVAL_IN}"

echo "==== eval cross-dataset (DROID-1p) + grounding probe ===="
python src/evaluate_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" \
  --fs_id "${RESULT_DIR}" --dataset_name droid --pov 1 --split test \
  --batch_size 1 --prediction_mode fusion --grounding_check \
  --result_folder "${EVAL_X}"

echo "Done post-merger ALF: ${RESULT_DIR} ${EVAL_IN} ${EVAL_X}"
