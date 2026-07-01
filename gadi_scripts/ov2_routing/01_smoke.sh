#!/bin/bash
#PBS -P ka69
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# GPU smoke test: 30-sample CALVIN-1p runs for (a) baseline head, (b) routed
# head, and (c) the MoE failure-expert router. Verifies the full forward/back
# path on a GPU before launching full training.
#
#   qsub gadi_scripts/ov2_routing/01_smoke.sh
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

COMMON="--vlm_model_id ${VLM_MODEL} --pov 1 --num_entry 30 --num_epochs 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 --pooling_mode tcond \
  --loss_mode fusion --prediction_mode fusion --batch_size 1"

echo "============================================================"
echo "(a) baseline head smoke (CALVIN)"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --result_folder ./results_ov2_smoke_baseline

echo "============================================================"
echo "(b) routed head smoke (CALVIN, dual-query, contrastive)"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_router --router_mode contrastive --gate_style multiplicative \
  --result_folder ./results_ov2_smoke_routed

echo "============================================================"
echo "(c) MoE failure-expert smoke (CALVIN, 4 experts, top-2)"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_moe --num_experts 4 --moe_top_k 2 --load_balance_coef 0.01 \
  --router_mode contrastive --gate_style multiplicative \
  --result_folder ./results_ov2_smoke_moe

echo "============================================================"
echo "(d) DROID routed smoke (exercises augment_droid_dataset balancing)"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name droid \
  --use_router --router_mode contrastive --gate_style multiplicative \
  --result_folder ./results_ov2_smoke_droid

echo "============================================================"
echo "(e) hierarchical routed + hybrid pooling smoke (CALVIN)"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_hier_router --router_mode contrastive --pooling_mode hybrid \
  --gate_style multiplicative \
  --result_folder ./results_ov2_smoke_hier_routed

echo "============================================================"
echo "(f) nested guided fusion smoke (CALVIN, text alpha + guiding + adapter)"
echo "     v2: inner residual h_l = V_l + beta_l * delta_l"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_nested_guided_fusion --ngf_layer_weight_mode text \
  --vision_layer_indices 9 17 24 --gate_style guiding \
  --use_merger_adapter --merger_adapter_rank 64 \
  --layer_balance_coef 0.01 \
  --result_folder ./results_ov2_smoke_ngf

echo "============================================================"
echo "(g) Arch B smoke: per-patch depth routing (token alpha_{l,n})"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_nested_guided_fusion --ngf_layer_weight_mode token \
  --vision_layer_indices 9 17 24 --gate_style guiding \
  --use_merger_adapter --merger_adapter_rank 64 \
  --layer_balance_coef 0.01 \
  --result_folder ./results_ov2_smoke_ngf_token

echo "============================================================"
echo "(h) Arch C smoke: sequential depth-recurrent refinement"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_nested_guided_fusion --ngf_sequential \
  --vision_layer_indices 9 17 24 --gate_style guiding \
  --use_merger_adapter --merger_adapter_rank 64 \
  --result_folder ./results_ov2_smoke_ngf_seq

echo "============================================================"
echo "(i) post-merger ALF smoke (CALVIN, fuse-after-merger + adapter)"
echo "============================================================"
python src/finetune_FS_ov2_routed.py $COMMON --dataset_name calvin \
  --use_post_merger_alf --vision_layer_indices 6 12 18 \
  --post_merger_adapter_rank 64 --alf_router_dim 256 \
  --result_folder ./results_ov2_smoke_post_merger_alf

echo "Smoke test complete."
