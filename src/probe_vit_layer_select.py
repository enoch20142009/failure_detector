"""Layer-selection diagnostic (Mysteries-of-the-Deep inspired).

Picks which intermediate ViT depths to fuse for the post-merger ALF model.
Reuses the Phase 0b text-guided probe machinery from ``probe_vit_modules``:
frozen ViT taps, frozen embed-table task/failure queries, fixed random
query->key projections for task-conditioned patch pooling, and a logistic
regression readout. No ViT/LM weights are trained.

For each candidate encoder depth ``l`` (indexed in the encoder ``hidden_states``
convention, matching ``--vision_layer_indices``), we score:

  * ``id_auroc``   - CALVIN failure separability (train on CALVIN train, eval on
                     CALVIN test) with the task-guided readout.
  * ``droid_auroc``- cross-domain proxy: same probe evaluated on balanced DROID,
                     mirroring the main CALVIN->DROID transfer metric.

A greedy complementary search then builds a set of ``K`` intermediates that
maximizes a transfer-weighted combined AUROC on the concatenated features. The
final recommendation feeds ``--vision_layer_indices`` for
``31_train_eval_post_merger_alf.sh`` (base index 24 is always kept as the anchor
and is therefore excluded from the candidate grid).

Example:
    python src/probe_vit_layer_select.py \\
        --vlm_model_id /scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct \\
        --pov 1 --num_entry 800 --select_k 3 \\
        --result_folder ./eval_results/ov2_probe_layer_select
"""

import argparse
import gc
import json
import os

import numpy as np
import torch

from load_dataset import augment_droid_dataset, load_data
from model_ov2_baseline import (
    DEFAULT_OV2_MODEL_ID,
    DEFAULT_OV2_REVISION,
    label_to_binary,
)
from probe_vit_modules import (
    FixedTextGuidedProjector,
    VisionFeatureExtractor,
    _fit_logreg,
    _predict_proba,
    roc_auc_score,
)

# Block output ("residual connection 2") = encoder hidden_states[block + 1].
# We probe this module because it is exactly what the model's block-tap fusion
# path (`--ngf_tap block` / `--vision_layer_indices`) consumes.
RC2 = "RC2"
BASE_INDEX = 24  # V_base / merger input (final block output); always the anchor.
DEFAULT_CANDIDATES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Greedy transfer-aware ViT layer selection for post-merger ALF."
    )
    parser.add_argument("--vlm_model_id", type=str, default=DEFAULT_OV2_MODEL_ID)
    parser.add_argument("--revision", type=str, default=DEFAULT_OV2_REVISION)
    parser.add_argument("--pov", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--style", type=str, default="image", choices=["image", "video"])
    parser.add_argument(
        "--num_entry",
        type=str,
        default="800",
        help="'full' or integer cap for CALVIN train (eval splits use full).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument(
        "--candidate_layers",
        type=int,
        nargs="*",
        default=DEFAULT_CANDIDATES,
        help="Candidate encoder hidden_states indices (exclude 24 = V_base anchor).",
    )
    parser.add_argument(
        "--select_k",
        type=int,
        default=3,
        help="Number of intermediate depths to recommend.",
    )
    parser.add_argument(
        "--text_pool_mode",
        type=str,
        default="concat",
        choices=["task", "fail", "contrast", "concat"],
    )
    parser.add_argument("--key_dim", type=int, default=256)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument(
        "--w_id", type=float, default=0.3, help="Weight on CALVIN in-domain AUROC."
    )
    parser.add_argument(
        "--w_xd", type=float, default=0.7, help="Weight on DROID cross-domain AUROC."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--result_folder",
        type=str,
        default="./eval_results/ov2_probe_layer_select",
    )
    return parser


def _parse_num_entry(num_entry):
    if num_entry in (None, "full"):
        return "full"
    return int(num_entry)


def extract_rc2_guided(extractor, dataset, split_name, projector, pool_mode, hidden_indices):
    """Task-guided RC2 features per candidate depth for one split.

    Returns ``feats[h] -> [N, D_feat]`` (keyed by hidden_states index) plus the
    binary label vector. Patches are pooled on the fly so full patch tensors are
    never retained for the whole split.
    """
    n = len(dataset)
    block_of = {h: h - 1 for h in hidden_indices}
    feats = {h: [] for h in hidden_indices}
    labels = []

    for i in range(n):
        sample = dataset[i]
        labels.append(label_to_binary(sample["label"]))
        patches = extractor.extract_patches(sample["images"])
        q_task, q_fail = extractor.compute_queries(sample["task"])
        for h in hidden_indices:
            vec = projector.guided_pool(
                patches[RC2][block_of[h]], q_task, q_fail, RC2, pool_mode
            )
            feats[h].append(vec)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{split_name}] {i + 1}/{n} samples", flush=True)

    feats = {h: np.stack(arrs, axis=0) for h, arrs in feats.items()}
    return feats, np.asarray(labels, dtype=np.int64)


def _fit_probe(x_train, y_train, max_iter, probe_c):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    w, b = _fit_logreg((x_train - mean) / std, y_train, max_iter, C=probe_c)
    return w, b, mean, std


def _subset_matrix(feats, subset):
    return np.concatenate([feats[h] for h in subset], axis=1).astype(np.float64)


def eval_subset(feats_tr, y_tr, feats_id, y_id, feats_xd, y_xd, subset, args):
    """Train one probe on CALVIN train, return (id_auroc, droid_auroc)."""
    x_tr = _subset_matrix(feats_tr, subset)
    w, b, mean, std = _fit_probe(x_tr, y_tr, args.max_iter, args.probe_C)
    id_auroc = roc_auc_score(
        y_id, _predict_proba(_subset_matrix(feats_id, subset), w, b, mean, std)
    )
    xd_auroc = roc_auc_score(
        y_xd, _predict_proba(_subset_matrix(feats_xd, subset), w, b, mean, std)
    )
    return id_auroc, xd_auroc


def _combined(id_auroc, xd_auroc, args):
    return args.w_id * id_auroc + args.w_xd * xd_auroc


def run_single_layer(feats_tr, y_tr, feats_id, y_id, feats_xd, y_xd, candidates, args):
    """Per-layer AUROCs + fitted single-layer probes (for correlation diag)."""
    scores = {}
    models = {}
    for h in candidates:
        x_tr = feats_tr[h].astype(np.float64)
        w, b, mean, std = _fit_probe(x_tr, y_tr, args.max_iter, args.probe_C)
        id_auroc = roc_auc_score(y_id, _predict_proba(feats_id[h], w, b, mean, std))
        xd_auroc = roc_auc_score(y_xd, _predict_proba(feats_xd[h], w, b, mean, std))
        scores[h] = {
            "id_auroc": id_auroc,
            "droid_auroc": xd_auroc,
            "combined": _combined(id_auroc, xd_auroc, args),
        }
        models[h] = (w, b, mean, std)
    return scores, models


def greedy_select(feats_tr, y_tr, feats_id, y_id, feats_xd, y_xd, candidates, args):
    """Greedy complementary selection maximizing combined concat-AUROC."""
    selected = []
    remaining = list(candidates)
    trace = []
    prev_combined = 0.0
    for step in range(min(args.select_k, len(candidates))):
        best_h, best = None, None
        for c in remaining:
            id_auroc, xd_auroc = eval_subset(
                feats_tr, y_tr, feats_id, y_id, feats_xd, y_xd, selected + [c], args
            )
            combined = _combined(id_auroc, xd_auroc, args)
            if best is None or combined > best["combined"]:
                best = {
                    "id_auroc": id_auroc,
                    "droid_auroc": xd_auroc,
                    "combined": combined,
                }
                best_h = c
        selected.append(best_h)
        remaining.remove(best_h)
        trace.append(
            {
                "step": step + 1,
                "added": best_h,
                "subset": list(selected),
                "id_auroc": best["id_auroc"],
                "droid_auroc": best["droid_auroc"],
                "combined": best["combined"],
                "gain": best["combined"] - prev_combined,
            }
        )
        prev_combined = best["combined"]
        print(
            f"  step {step + 1}: +layer {best_h} -> subset {selected} "
            f"id={best['id_auroc']:.4f} droid={best['droid_auroc']:.4f} "
            f"combined={best['combined']:.4f} (gain {trace[-1]['gain']:+.4f})",
            flush=True,
        )
    return selected, trace


def pairwise_score_correlation(feats_id, models, candidates):
    """Corr of per-sample single-layer fail-scores (Mysteries-style diagnostic)."""
    prob_cols = []
    for h in candidates:
        w, b, mean, std = models[h]
        prob_cols.append(_predict_proba(feats_id[h], w, b, mean, std))
    probs = np.stack(prob_cols, axis=1)  # [N, L]
    if probs.shape[0] < 2:
        return np.eye(len(candidates)).tolist(), float("nan")
    corr = np.corrcoef(probs, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    # Disagreement = mean per-sample std of layer fail-scores (higher = layers
    # carry more distinct signal); a lightweight training-free entropy proxy.
    disagreement = float(probs.std(axis=1).mean())
    return corr.tolist(), disagreement


def print_single_layer_table(scores, candidates):
    print("\n==== SINGLE-LAYER AUROC (RC2, text-guided) ====")
    header = "layer |  id_auroc | droid_auroc |  combined"
    print(header)
    print("-" * len(header))
    for h in candidates:
        s = scores[h]
        print(
            f"{h:>5} |   {s['id_auroc']:.4f} |     {s['droid_auroc']:.4f} |   "
            f"{s['combined']:.4f}"
        )


def main():
    args = build_parser().parse_args()
    os.makedirs(args.result_folder, exist_ok=True)
    num_entry = _parse_num_entry(args.num_entry)

    candidates = [h for h in args.candidate_layers if h != BASE_INDEX]
    if not candidates:
        raise ValueError("No candidate layers left after excluding the base index 24.")

    print("Loading splits...")
    calvin_train = load_data(
        dataset_name="calvin", style=args.style, pov=args.pov, split="train",
        num_entry=num_entry, seed=args.seed,
    )
    calvin_test = load_data(
        dataset_name="calvin", style=args.style, pov=args.pov, split="test",
        num_entry="full", seed=args.seed,
    )
    droid_raw = load_data(
        dataset_name="droid", style=args.style, pov=args.pov, split="test",
        num_entry="full", seed=args.seed,
    )
    # Raw DROID is single-class (all success); balance it to get a real
    # cross-domain failure AUROC, matching the training/eval pipeline.
    droid_bal = augment_droid_dataset(droid_raw)
    print(f"Balanced DROID eval split: {len(droid_bal)} samples")

    print("Building feature extractor (vision forward only)...")
    block_indices = [h - 1 for h in candidates]
    extractor = VisionFeatureExtractor(
        model_id=args.vlm_model_id,
        revision=args.revision,
        device=args.device,
        max_pixels=args.max_pixels,
        layers=block_indices,
    )
    projector = FixedTextGuidedProjector(
        extractor.module_dims, extractor.query_dim, args.key_dim, args.seed
    )
    print(
        f"Candidate hidden_states indices: {candidates} "
        f"(encoder blocks {block_indices}); base anchor = {BASE_INDEX}"
    )

    print("Extracting CALVIN train...")
    feats_tr, y_tr = extract_rc2_guided(
        extractor, calvin_train, "calvin-train", projector, args.text_pool_mode, candidates
    )
    print("Extracting CALVIN test...")
    feats_id, y_id = extract_rc2_guided(
        extractor, calvin_test, "calvin-test", projector, args.text_pool_mode, candidates
    )
    print("Extracting balanced DROID...")
    feats_xd, y_xd = extract_rc2_guided(
        extractor, droid_bal, "droid", projector, args.text_pool_mode, candidates
    )
    extractor.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nScoring single layers...")
    single_scores, single_models = run_single_layer(
        feats_tr, y_tr, feats_id, y_id, feats_xd, y_xd, candidates, args
    )
    print_single_layer_table(single_scores, candidates)

    print("\nGreedy complementary selection...")
    selected, trace = greedy_select(
        feats_tr, y_tr, feats_id, y_id, feats_xd, y_xd, candidates, args
    )
    pairwise_corr, disagreement = pairwise_score_correlation(
        feats_id, single_models, candidates
    )

    recommended = sorted(selected)
    out = {
        "recommended_intermediate_indices": recommended,
        "base_index": BASE_INDEX,
        "select_k": args.select_k,
        "candidate_layers": candidates,
        "score_weights": {"w_id": args.w_id, "w_xd": args.w_xd},
        "single_layer_scores": {str(h): single_scores[h] for h in candidates},
        "greedy_trace": trace,
        "greedy_order": selected,
        "pairwise_score_corr": pairwise_corr,
        "layer_score_disagreement": disagreement,
        "config": vars(args),
    }
    out_path = os.path.join(args.result_folder, "layer_select.json")
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=2)

    print("\n" + "=" * 60)
    print(f"Recommended intermediate indices (greedy K={args.select_k}): {recommended}")
    print(f"Mean layer-score disagreement (higher=more distinct): {disagreement:.4f}")
    print("Copy-paste for the training script:")
    print(f"  --vision_layer_indices {' '.join(str(h) for h in recommended)}")
    print(f"\nSaved layer selection to {out_path}")


if __name__ == "__main__":
    main()
