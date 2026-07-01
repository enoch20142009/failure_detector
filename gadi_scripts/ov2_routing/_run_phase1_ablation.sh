#!/bin/bash
# Shared train+eval driver for Phase 1 failure isolation ablations (A–E).
# Usage (from repo root, venv active):
#   bash gadi_scripts/ov2_routing/_run_phase1_ablation.sh A
#
set -euo pipefail

ABL="${1:?Usage: _run_phase1_ablation.sh A|B|C|D|E}"

VLM_MODEL="${VLM_MODEL:-/scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct}"

# Shared hyperparams (match Phase 1 / NGF v2 unless ablation overrides).
COMMON=(
  --vlm_model_id "${VLM_MODEL}"
  --dataset_name calvin --pov 1
  --num_epochs 5 --batch_size 1
  --target_layer_indices 19 28 36 --num_classifiers 3
  --pooling_mode tcond --loss_mode fusion --prediction_mode fusion
  --use_nested_guided_fusion --ngf_layer_weight_mode text
  --vision_layer_indices 6 12 18
  --router_mode contrastive --gate_type sigmoid --gate_style guiding
  --layer_balance_coef 0.01
  --dropout_rate 0.1 --lr 1e-4 --weight_decay 0.1
)

case "${ABL}" in
  A)
    # Phase 1 + V_base back: is missing base the killer?
    TAG="p1_ablate_A_base_back"
    EXTRA=(--ngf_tap ffn_act --ngf_full_connector)
    ;;
  B)
    # FFN-Act + intermediate-only + merger adapter (no full connector).
    TAG="p1_ablate_B_ffn_act_adapter"
    EXTRA=(--ngf_tap ffn_act --ngf_intermediate_only --use_merger_adapter --merger_adapter_rank 64)
    ;;
  C)
    # Block tap + intermediate-only + full connector: is FFN-Act/down-proj the killer?
    TAG="p1_ablate_C_block_fullconn"
    EXTRA=(--ngf_tap block --ngf_intermediate_only --ngf_full_connector)
    ;;
  D)
    # Sanitized intermediate: FFN-Act + base + adapter (drop aggressive choices).
    TAG="p1_ablate_D_sanitized"
    EXTRA=(--ngf_tap ffn_act --use_merger_adapter --merger_adapter_rank 64)
    ;;
  E)
    # Working NGF v2 stack; only layer indices changed to {6,12,18}.
    TAG="p1_ablate_E_ngf_v2_layers618"
    EXTRA=(--ngf_tap block --use_merger_adapter --merger_adapter_rank 64)
    ;;
  *)
    echo "Unknown ablation '${ABL}'. Expected A, B, C, D, or E." >&2
    exit 1
    ;;
esac

RESULT_DIR="./results_ov2_calvin_${TAG}"
EVAL_IN="./eval_results/ov2_calvin_${TAG}_indomain"
EVAL_X="./eval_results/ov2_calvin_${TAG}_to_droid"

echo "==== Phase 1 ablation ${ABL}: ${TAG} ===="
echo "RESULT_DIR=${RESULT_DIR}"

python src/finetune_FS_ov2_routed.py \
  "${COMMON[@]}" \
  "${EXTRA[@]}" \
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

echo "Done ablation ${ABL}: ${RESULT_DIR} ${EVAL_IN} ${EVAL_X}"
