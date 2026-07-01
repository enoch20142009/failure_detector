#!/bin/bash
#PBS -P ka69
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l walltime=12:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -j oe
#
# Ablation E — NGF v2 working stack (block tap + base + adapter) at layers {6,12,18}.
# Tests: do intermediate block taps help when the rest of the stack is known-good?
# (NGF v2 at 9/17/24 AUROC ~0.792; champion flat router ~0.832 has no layer hooks.)
#
#   qsub gadi_scripts/ov2_routing/27_phase1_ablation_E_ngf_v2_layers618.sh
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
bash gadi_scripts/ov2_routing/_run_phase1_ablation.sh E
