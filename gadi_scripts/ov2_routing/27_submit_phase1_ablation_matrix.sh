#!/bin/bash
# Submit all Phase 1 isolation ablations (A–E) to gpuhopper.
#
#   bash gadi_scripts/ov2_routing/27_submit_phase1_ablation_matrix.sh
#   bash gadi_scripts/ov2_routing/27_submit_phase1_ablation_matrix.sh D   # single ablation
#
# Estimated cost: ~5 × ~312 SU ≈ 1560 SU total (same walltime as Phase 1 each).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

submit_one() {
  local letter="$1"
  local script
  case "${letter}" in
    A) script="gadi_scripts/ov2_routing/27_phase1_ablation_A_base_back.sh" ;;
    B) script="gadi_scripts/ov2_routing/27_phase1_ablation_B_ffn_act_adapter.sh" ;;
    C) script="gadi_scripts/ov2_routing/27_phase1_ablation_C_block_fullconn.sh" ;;
    D) script="gadi_scripts/ov2_routing/27_phase1_ablation_D_sanitized.sh" ;;
    E) script="gadi_scripts/ov2_routing/27_phase1_ablation_E_ngf_v2_layers618.sh" ;;
    *) echo "Unknown ablation '${letter}'" >&2; return 1 ;;
  esac
  echo "Submitting ablation ${letter}: ${script}"
  qsub "${script}"
}

if [[ $# -eq 0 ]]; then
  for letter in A B C D E; do
    submit_one "${letter}"
  done
else
  for letter in "$@"; do
    submit_one "${letter}"
  done
fi

echo ""
echo "After jobs finish, collect with:"
echo "  python gadi_scripts/ov2_routing/collect_phase1_ablations.py"
