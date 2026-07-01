#!/bin/bash
# Submit multi-seed + beta-reg overnight batch.
#
#   bash gadi_scripts/ov2_routing/28_submit_multiseed_overnight.sh
#
# Jobs: M0 champion s123/s456 (gpuhopper), M1 s123/s456 + breg01 (dgxa100).
# Estimated: 5 × ~312 SU ≈ 1560 SU.
#
set -euo pipefail

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main

chmod +x gadi_scripts/ov2_routing/12_train_eval_calvin_routed_merger_adapter_s123.sh
chmod +x gadi_scripts/ov2_routing/12_train_eval_calvin_routed_merger_adapter_s456.sh
chmod +x gadi_scripts/ov2_routing/17_ablate_inner_off_s123.sh
chmod +x gadi_scripts/ov2_routing/17_ablate_inner_off_s456.sh

echo "Submitting M0 champion multi-seed..."
qsub gadi_scripts/ov2_routing/12_train_eval_calvin_routed_merger_adapter_s123.sh
qsub gadi_scripts/ov2_routing/12_train_eval_calvin_routed_merger_adapter_s456.sh

echo "Submitting M1 A2 inner-off multi-seed..."
qsub gadi_scripts/ov2_routing/17_ablate_inner_off_s123.sh
qsub gadi_scripts/ov2_routing/17_ablate_inner_off_s456.sh

echo "Submitting NGF beta-reg L2=0.01..."
qsub gadi_scripts/ov2_routing/21_ngf_v2_breg01.sh

echo ""
echo "Submitted 5 overnight jobs. Monitor: qstat -u $USER"
