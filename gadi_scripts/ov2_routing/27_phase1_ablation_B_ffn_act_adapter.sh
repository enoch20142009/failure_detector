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
# Ablation B — FFN-Act + intermediate-only + merger adapter (no full connector).
# Tests: is replacing the frozen merger the killer?
#
#   qsub gadi_scripts/ov2_routing/27_phase1_ablation_B_ffn_act_adapter.sh
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
bash gadi_scripts/ov2_routing/_run_phase1_ablation.sh B
