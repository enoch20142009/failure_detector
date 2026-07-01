#!/usr/bin/env python3
"""Collect OV2 routed/MoE eval results into a single comparison table.

Scans ``eval_results/ov2_*`` for ``results.json`` files written by
``evaluate_FS_ov2_routed.py`` and prints a markdown table covering the full
baseline / routed / MoE x CALVIN / DROID matrix (in-domain and cross-dataset).

Usage:
    python gadi_scripts/ov2_routing/collect_results.py
"""

import json
import os
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(ROOT, "eval_results")

# Expected eval folders for the 6-cell matrix (in-domain + cross-dataset).
EXPECTED = [
    "ov2_calvin_baseline_indomain",
    "ov2_calvin_baseline_to_droid",
    "ov2_droid_baseline_indomain",
    "ov2_droid_baseline_to_calvin",
    "ov2_calvin_routed_indomain",
    "ov2_calvin_routed_to_droid",
    "ov2_droid_routed_indomain",
    "ov2_droid_routed_to_calvin",
    "ov2_calvin_moe_indomain",
    "ov2_calvin_moe_to_droid",
    "ov2_droid_moe_indomain",
    "ov2_droid_moe_to_calvin",
    "ov2_calvin_hier_routed_indomain",
    "ov2_calvin_hier_routed_to_droid",
    "ov2_calvin_routed_merger_adapter_indomain",
    "ov2_calvin_routed_merger_adapter_to_droid",
    "ov2_calvin_routed_hybrid_indomain",
    "ov2_calvin_routed_hybrid_to_droid",
    "ov2_calvin_fuse_then_route_merger_adapter_indomain",
    "ov2_calvin_fuse_then_route_merger_adapter_to_droid",
    "ov2_calvin_fuse_then_route_merger_adapter_guiding_indomain",
    "ov2_calvin_fuse_then_route_merger_adapter_guiding_to_droid",
    "ov2_calvin_m3_text_fuse_route_indomain",
    "ov2_calvin_m3_text_fuse_route_to_droid",
    "ov2_calvin_tgif_merger_adapter_indomain",
    "ov2_calvin_tgif_merger_adapter_to_droid",
    # Nested Guided Fusion (NGF-0 primary + A1-A8 ablation ladder).
    "ov2_calvin_ngf_indomain",
    "ov2_calvin_ngf_to_droid",
    "ov2_calvin_ngf_a1_uniform_indomain",
    "ov2_calvin_ngf_a1_uniform_to_droid",
    "ov2_calvin_ngf_a2_inner_off_indomain",
    "ov2_calvin_ngf_a2_inner_off_to_droid",
    "ov2_calvin_ngf_a3_static_indomain",
    "ov2_calvin_ngf_a3_static_to_droid",
    "ov2_calvin_ngf_a4_no_adapter_indomain",
    "ov2_calvin_ngf_a4_no_adapter_to_droid",
    "ov2_calvin_ngf_a5_residual_indomain",
    "ov2_calvin_ngf_a5_residual_to_droid",
    "ov2_calvin_ngf_a6_mult_indomain",
    "ov2_calvin_ngf_a6_mult_to_droid",
    "ov2_calvin_ngf_a7_single_indomain",
    "ov2_calvin_ngf_a7_single_to_droid",
    "ov2_calvin_ngf_a8_hybrid_indomain",
    "ov2_calvin_ngf_a8_hybrid_to_droid",
    # Phase 1 intermediate FFN-Act (failed) + isolation ablations A–E.
    "ov2_calvin_ngf_ffn_act_intermediate_indomain",
    "ov2_calvin_ngf_ffn_act_intermediate_to_droid",
    "ov2_calvin_p1_ablate_A_base_back_indomain",
    "ov2_calvin_p1_ablate_A_base_back_to_droid",
    "ov2_calvin_p1_ablate_B_ffn_act_adapter_indomain",
    "ov2_calvin_p1_ablate_B_ffn_act_adapter_to_droid",
    "ov2_calvin_p1_ablate_C_block_fullconn_indomain",
    "ov2_calvin_p1_ablate_C_block_fullconn_to_droid",
    "ov2_calvin_p1_ablate_D_sanitized_indomain",
    "ov2_calvin_p1_ablate_D_sanitized_to_droid",
    "ov2_calvin_p1_ablate_E_ngf_v2_layers618_indomain",
    "ov2_calvin_p1_ablate_E_ngf_v2_layers618_to_droid",
]

FIELDS = ["accuracy", "precision", "recall", "f1", "auroc", "average_precision", "ece", "brier"]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def load_one(folder):
    path = os.path.join(EVAL_DIR, folder, "results.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def main():
    print("# OV2 Results Collection\n")
    header = "| run | n | " + " | ".join(FIELDS) + " | extra |"
    sep = "| --- " * (len(FIELDS) + 3) + "|"
    print(header)
    print(sep)

    for folder in EXPECTED:
        data = load_one(folder)
        if data is None:
            print(f"| {folder} | PENDING | " + " | ".join(["-"] * len(FIELDS)) + " | - |")
            continue
        row = [folder, str(data.get("num_samples", "-"))]
        for field in FIELDS:
            row.append(fmt(data.get(field)))
        extra = []
        if "grounding_flip_rate" in data:
            extra.append(f"flip={fmt(data['grounding_flip_rate'])}")
        if "expert_usage_hist" in data:
            extra.append(f"experts={data['expert_usage_hist']}")
        if "mean_gate_entropy_nats" in data:
            extra.append(f"gate_H={fmt(data['mean_gate_entropy_nats'])}")
        if "vision_fusion_weights" in data:
            extra.append(f"vfusion={data['vision_fusion_weights']}")
        if "ngf_layer_alpha" in data:
            extra.append(f"alpha={data['ngf_layer_alpha']}")
        if "ngf_mean_patch_gate_entropy" in data:
            extra.append(f"ngf_gate_H={fmt(data['ngf_mean_patch_gate_entropy'])}")
        if "confusion_matrix" in data:
            extra.append(f"cm={data['confusion_matrix']}")
        row.append("; ".join(extra) if extra else "-")
        print("| " + " | ".join(row) + " |")

    # Surface any unexpected folders too.
    found = {os.path.basename(os.path.dirname(p)) for p in glob(os.path.join(EVAL_DIR, "ov2_*", "results.json"))}
    extra_folders = sorted(found - set(EXPECTED))
    if extra_folders:
        print("\nOther ov2_* result folders found:")
        for f in extra_folders:
            print(f"- {f}")


if __name__ == "__main__":
    main()
