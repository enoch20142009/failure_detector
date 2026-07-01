#!/usr/bin/env python3
"""Compare Phase 1 failure vs isolation ablations A–E (+ reference runs).

Reads ``eval_results/ov2_calvin_*_to_droid/results.json`` (OOD primary metric)
and in-domain folders for completeness. Prints a decision matrix for which
Phase 1 design choice caused collapse.

Usage:
    python gadi_scripts/ov2_routing/collect_phase1_ablations.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(ROOT, "eval_results")

# Rows in causal isolation order. Phase 1 + references for context.
ROWS = [
    ("Phase 1 (failed)", "ov2_calvin_ngf_ffn_act_intermediate", "ffn_act", "yes", "yes", "no", "no"),
    ("A: +V_base", "ov2_calvin_p1_ablate_A_base_back", "ffn_act", "no", "yes", "no", "no"),
    ("B: +adapter", "ov2_calvin_p1_ablate_B_ffn_act_adapter", "ffn_act", "yes", "no", "yes", "no"),
    ("C: block tap", "ov2_calvin_p1_ablate_C_block_fullconn", "block", "yes", "yes", "no", "no"),
    ("D: sanitized", "ov2_calvin_p1_ablate_D_sanitized", "ffn_act", "no", "no", "yes", "no"),
    ("E: NGF v2 @618", "ov2_calvin_p1_ablate_E_ngf_v2_layers618", "block", "no", "no", "yes", "no"),
    ("— ref: champion", "ov2_calvin_routed_merger_adapter", "—", "—", "—", "yes", "—"),
    ("— ref: NGF v2 @924", "ov2_calvin_ngf_v2", "block", "no", "no", "yes", "—"),
]


def load(folder, split):
    path = os.path.join(EVAL, f"{folder}_{split}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    print("# Phase 1 isolation ablation matrix\n")
    print("Primary OOD metric: **CALVIN→DROID AUROC**. Grounding flip > 0 ⇒ task-sensitive routing.\n")

    header = (
        "| run | tap | inter_only | full_conn | adapter | "
        "OOD AUROC | ID AUROC | flip | all-fail? | alpha | ngf_gate_H |"
    )
    print(header)
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for label, base, tap, inter, full, adapter, _ in ROWS:
        ood = load(base, "to_droid")
        ind = load(base, "indomain")

        def g(d, k):
            return d.get(k) if d else None

        alpha = g(ood, "ngf_layer_alpha") or g(ind, "ngf_layer_alpha")
        alpha_s = fmt(alpha[0], 3) if isinstance(alpha, list) and alpha else fmt(alpha)

        cm = g(ood, "confusion_matrix")
        all_fail = "yes" if cm and len(cm) == 2 and cm[1][1] == 0 else ("no" if cm else "—")

        print(
            f"| {label} | {tap} | {inter} | {full} | {adapter} | "
            f"{fmt(g(ood, 'auroc'))} | {fmt(g(ind, 'auroc'))} | "
            f"{fmt(g(ood, 'grounding_flip_rate'))} | {all_fail} | "
            f"{alpha_s} | {fmt(g(ood, 'ngf_mean_patch_gate_entropy'), 2)} |"
        )

    print("\n## How to read\n")
    print("- **A vs Phase 1**: if A recovers AUROC/flip, missing `V_base` was a major cause.")
    print("- **B vs Phase 1**: if B recovers, full connector (replace merger) was a major cause.")
    print("- **C vs Phase 1**: if C >> Phase 1, FFN-Act/down-proj was a major cause.")
    print("- **D**: best honest intermediate experiment if any FFN-Act signal exists.")
    print("- **E vs NGF v2 @924**: if E ≈ v2, layers 6/12/18 are fine; if E << v2, depth choice matters.")
    print("\nPending rows show `—` until eval ``results.json`` exists.\n")


if __name__ == "__main__":
    main()
