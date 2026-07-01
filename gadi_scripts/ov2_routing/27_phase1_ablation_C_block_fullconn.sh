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
# Ablation C — Block tap {6,12,18} + intermediate-only + full connector.
# Tests: is FFN-Act / random down-proj the killer (vs block hidden states)?
#
#   qsub gadi_scripts/ov2_routing/27_phase1_ablation_C_block_fullconn.sh
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
bash gadi_scripts/ov2_routing/_run_phase1_ablation.sh C
