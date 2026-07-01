#!/bin/bash
# Submit the NGF v2 follow-up batch from the Gadi login node.
# 2 Arch C multi-seed jobs (seed 123, 2024) + 2 beta-regularized NGF-0 jobs
# (L2=0.01, 0.05). All dgxa100, ngpus=1 ncpus=16.
#
#   bash gadi_scripts/ov2_routing/22_submit_followup_ngf_v2.sh
#
set -euo pipefail

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main

# Arch C multi-seed (combine with existing seed=42 run)
qsub gadi_scripts/ov2_routing/20_ngf_v2c_seq_s123.sh
qsub gadi_scripts/ov2_routing/20_ngf_v2c_seq_s2024.sh

# beta-regularized NGF-0 (capacity sweep vs A2 transfer)
qsub gadi_scripts/ov2_routing/21_ngf_v2_breg01.sh
qsub gadi_scripts/ov2_routing/21_ngf_v2_breg05.sh

echo "Submitted 4 NGF v2 follow-up jobs (all dgxa100). Monitor: qstat -u $USER"
