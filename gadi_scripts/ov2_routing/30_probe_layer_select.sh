#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=50GB
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Layer-selection diagnostic for the post-merger ALF model.
#   Greedy transfer-aware selection of intermediate ViT depths (Mysteries-of-
#   the-Deep inspired). Writes recommended --vision_layer_indices to
#   eval_results/ov2_probe_layer_select/layer_select.json for use by
#   31_train_eval_post_merger_alf.sh.
#
#   Streams one sample at a time (no full-split patch store), mirroring the
#   Phase 0b probe memory profile (128GB).
#
#   qsub gadi_scripts/ov2_routing/30_probe_layer_select.sh
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
RESULT_DIR=./eval_results/ov2_probe_layer_select

echo "==== Layer-selection diagnostic (greedy transfer-aware, dgxa100) ===="
python src/probe_vit_layer_select.py \
  --vlm_model_id "${VLM_MODEL}" \
  --pov 1 --style image \
  --num_entry 800 \
  --max_pixels 200704 \
  --candidate_layers 4 6 8 10 12 14 16 18 20 22 \
  --select_k 3 \
  --w_id 0.3 --w_xd 0.7 \
  --result_folder "${RESULT_DIR}"

echo "Done: ${RESULT_DIR}/layer_select.json"
