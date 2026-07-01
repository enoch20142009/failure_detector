#!/bin/bash
# Submit the NGF v2 overnight batch from the Gadi login node.
# 4 gpuhopper jobs (v2 primary + ablations A1/A5/A6) + 2 dgxa100 novel archs (B/C).
#
#   bash gadi_scripts/ov2_routing/18_submit_overnight_ngf_v2.sh
#
set -euo pipefail

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main

# gpuhopper: v2 primary + ablations
qsub gadi_scripts/ov2_routing/17_train_eval_calvin_nested_guided_fusion.sh
qsub gadi_scripts/ov2_routing/17_ablate_inner_uniform_alpha.sh
qsub gadi_scripts/ov2_routing/17_ablate_residual_base.sh
qsub gadi_scripts/ov2_routing/17_ablate_mult_gates.sh

# dgxa100: novel architectures
qsub gadi_scripts/ov2_routing/19_ngf_v2b_token_depth.sh
qsub gadi_scripts/ov2_routing/19_ngf_v2c_sequential.sh

echo "Submitted 6 NGF v2 jobs (4 gpuhopper + 2 dgxa100). Monitor: qstat -u $USER"
