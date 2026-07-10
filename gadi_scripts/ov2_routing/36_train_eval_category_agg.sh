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
# Category contrastive aggregator (6×4 ViT categories).
#
# Env overrides:
#   CATEGORY_CONCAT_MODE=residual_last|igva_penultimate|igva_base  (default residual_last)
#   FULL_CONNECTOR=1   trainable merger clone, no MergerAdapter
#   FULL_CONNECTOR=0   frozen merger + MergerAdapter (default)
#
# Examples:
#   qsub gadi_scripts/ov2_routing/36_train_eval_category_agg.sh
#   CATEGORY_CONCAT_MODE=igva_base qsub gadi_scripts/ov2_routing/36_train_eval_category_agg.sh
#   CATEGORY_CONCAT_MODE=residual_last FULL_CONNECTOR=1 qsub gadi_scripts/ov2_routing/36_train_eval_category_agg.sh
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
CONCAT_MODE="${CATEGORY_CONCAT_MODE:-residual_last}"
FULL_CONNECTOR="${FULL_CONNECTOR:-0}"

case "${CONCAT_MODE}" in
  residual_last)
    TAG="category_agg"
    if [[ "${FULL_CONNECTOR}" == "1" ]]; then TAG="category_agg_full_connector"; fi
    ;;
  igva_penultimate) TAG="category_agg_igva" ;;
  igva_base)        TAG="category_agg_igva_base" ;;
  *)
    echo "Unknown CATEGORY_CONCAT_MODE=${CONCAT_MODE}"
    exit 1
    ;;
esac

RESULT_DIR="./results_ov2_calvin_${TAG}"
EVAL_IN="./eval_results/ov2_calvin_${TAG}_indomain"
EVAL_X="./eval_results/ov2_calvin_${TAG}_to_droid"

MERGER_FLAGS=(--use_merger_adapter --merger_adapter_rank 64)
CONNECTOR_FLAG=()
if [[ "${FULL_CONNECTOR}" == "1" ]]; then
  MERGER_FLAGS=()
  CONNECTOR_FLAG=(--ngf_full_connector)
fi

echo "==== Category agg: concat_mode=${CONCAT_MODE} full_connector=${FULL_CONNECTOR} ===="
echo "RESULT_DIR=${RESULT_DIR}"

python src/finetune_FS_ov2_routed.py \
  --vlm_model_id "${VLM_MODEL}" --dataset_name calvin --pov 1 \
  --num_epochs 5 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion \
  --use_category_aggregator \
  --category_concat_mode "${CONCAT_MODE}" \
  --category_adapter_rank 256 \
  "${MERGER_FLAGS[@]}" \
  "${CONNECTOR_FLAG[@]}" \
  --layer_balance_coef 0.01 --router_dim 256 \
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

python3 - <<PY
import json
for name, path in [("CALVIN", "${EVAL_IN}/results.json"), ("DROID", "${EVAL_X}/results.json")]:
    d = json.load(open(path))
    print(f"{name}: auroc={d['auroc']:.4f} acc={d['accuracy']:.4f} "
          f"concat={d.get('category_concat_mode','?')}")
PY

echo "Done category agg: ${RESULT_DIR}"
