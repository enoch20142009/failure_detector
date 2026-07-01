"""Evaluation driver for the OneVision-2 routed MaTCA model (with optional MoE).

Loads a trained checkpoint, reconstructs the model configuration (including the
Stage-1 routing / MoE / gate-style settings) from the saved checkpoint and the
``config.txt`` written at train time, runs binary success/failure prediction on
a chosen split, and reports:

  - classification metrics: accuracy, precision, recall, F1, confusion matrix;
  - probabilistic metrics: AUROC, average precision, ECE, Brier score;
  - routing diagnostics: residual scales (alpha/beta), layer-fusion weights;
  - MoE diagnostics (when enabled): per-expert usage histogram, mean gate entropy;
  - grounding-leakage probe (``--grounding_check``): how often the prediction
    flips and the mean probability shift when the QUERY TEXT is swapped to an
    unrelated task while the IMAGE is held fixed. Because the routed values are
    purely visual (text only gates), this measures legitimate query sensitivity
    rather than text content leaking into the residual.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from load_dataset import augment_droid_dataset, load_data
from model_ov2_baseline import label_to_binary
from model_ov2_routed_matca import OV2RoutedMaTCA


# Boolean / list flags whose string forms must be parsed back from config.txt.
_BOOL_KEYS = {
    "use_hier_fusion", "use_tgif_fusion", "use_router", "use_fuse_then_route",
    "use_hier_router", "use_moe", "share_query", "moe_gate_supervision", "use_merger_adapter",
    "use_nested_guided_fusion", "ngf_inner_guiding", "nested_replace_base",
    "use_ngf_sequential", "use_post_merger_alf", "post_merger_adapter",
}
_INT_LIST_KEYS = {"target_layer_indices", "vision_layer_indices"}


def parse_config_txt(path):
    """Parse the ``key=value`` config.txt written by the training driver."""
    cfg = {}
    if not os.path.isfile(path):
        return cfg
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    return cfg


def infer_checkpoint_config(checkpoint_path, config_txt):
    """Merge the checkpoint's stored config with config.txt (checkpoint wins)."""
    cfg = dict(config_txt)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in (
        "num_classifiers", "dropout_rate", "target_layer_indices", "vision_layer_indices",
        "pooling_mode", "fusion_mode", "use_hier_fusion", "use_tgif_fusion", "use_router",
        "use_fuse_then_route", "use_hier_router", "depth_fusion_mode", "router_mode",
        "gate_type", "gate_style", "use_moe", "num_experts", "moe_top_k", "load_balance_coef",
        "use_merger_adapter", "merger_adapter_rank",
        "use_nested_guided_fusion", "ngf_layer_weight_mode", "ngf_inner_guiding",
        "nested_replace_base", "use_ngf_sequential", "layer_balance_coef",
        "ngf_tap", "ngf_intermediate_only", "ngf_full_connector",
        "use_post_merger_alf", "post_merger_adapter", "post_merger_adapter_rank",
        "alf_router_dim",
    ):
        if key in ckpt:
            cfg[key] = ckpt[key]
    return cfg


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes")


def _as_int_list(value, default):
    if isinstance(value, list):
        return value
    if value is None:
        return default
    cleaned = str(value).strip().strip("[]")
    return [int(tok) for tok in cleaned.replace(",", " ").split() if tok]


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.mean()) * abs(conf - acc)
    return float(ece)


def gate_entropy_from_logits(logits):
    """Mean per-token entropy (nats) of the softmax gate distribution."""
    p = torch.softmax(logits, dim=-1)
    ent = -(p * torch.clamp(p, min=1e-9).log()).sum(dim=-1)
    return float(ent.mean().item())


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate OneVision-2 routed MaTCA.")
    parser.add_argument("--vlm_model_id", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--fs_id", type=str, required=True,
                        help="Folder with the trained components + config.txt.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename inside fs_id (default: latest 'best').")
    parser.add_argument("--dataset_name", type=str, default="calvin",
                        choices=["calvin", "droid", "aha"])
    parser.add_argument("--pov", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--style", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--num_entry", type=str, default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prediction_mode", type=str, default="fusion",
                        choices=["fusion", "head_average", "head_majority"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_folder", type=str, default="./eval_results/ov2_routed")
    # Diagnostics / interventions.
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable synthetic class balancing for DROID "
                             "(by default DROID eval is balanced via "
                             "augment_droid_dataset, matching the Qwen pipeline).")
    parser.add_argument("--grounding_check", action="store_true",
                        help="Run the query-swap grounding-leakage probe.")
    parser.add_argument("--task_substitution", type=str, default=None,
                        help="Replace every task string with this fixed text (intervention).")
    return parser


def _find_checkpoint(fs_id, explicit):
    if explicit is not None:
        path = explicit if os.path.isabs(explicit) else os.path.join(fs_id, explicit)
        return path
    candidates = [f for f in os.listdir(fs_id) if f.endswith(".pt")]
    best = [f for f in candidates if "best" in f]
    pool = best or candidates
    if not pool:
        raise FileNotFoundError(f"No .pt checkpoint found in {fs_id}")
    pool.sort()
    return os.path.join(fs_id, pool[-1])


def main():
    args = build_parser().parse_args()
    os.makedirs(args.result_folder, exist_ok=True)

    config_txt = parse_config_txt(os.path.join(args.fs_id, "config.txt"))
    checkpoint_path = _find_checkpoint(args.fs_id, args.checkpoint)
    cfg = infer_checkpoint_config(checkpoint_path, config_txt)
    print(f"Using checkpoint: {checkpoint_path}")

    num_entry = args.num_entry
    if num_entry != "full":
        num_entry = int(num_entry)

    dataset = load_data(
        dataset_name=args.dataset_name,
        style=args.style,
        pov=args.pov,
        split=args.split,
        num_entry=num_entry,
        seed=args.seed,
    )

    # DROID HF splits are single-class (all success); synthesize a balanced
    # failure half via task-shift augmentation so cross-dataset and in-domain
    # DROID metrics are meaningful (matches the Qwen pipeline).
    if args.dataset_name == "droid" and not args.no_augmentation:
        print(f"Augmenting DROID eval split ({len(dataset)} -> balanced)...")
        dataset = augment_droid_dataset(dataset)
        print(f"  eval now {len(dataset)} samples")

    model_kwargs = dict(
        device=args.device,
        max_pixels=args.max_pixels,
        num_classifiers=int(cfg.get("num_classifiers", 3)),
        target_layer_indices=_as_int_list(cfg.get("target_layer_indices"), [19, 28, 36]),
        pooling_mode=cfg.get("pooling_mode", "tcond"),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        use_hier_fusion=_as_bool(cfg.get("use_hier_fusion")),
        use_tgif_fusion=_as_bool(cfg.get("use_tgif_fusion")),
        use_fuse_then_route=_as_bool(cfg.get("use_fuse_then_route")),
        use_router=_as_bool(cfg.get("use_router")),
        use_hier_router=_as_bool(cfg.get("use_hier_router")),
        depth_fusion_mode=cfg.get("depth_fusion_mode", "query_cond"),
        router_mode=cfg.get("router_mode", "contrastive"),
        gate_type=cfg.get("gate_type", "sigmoid"),
        gate_style=cfg.get("gate_style", "multiplicative"),
        share_query=_as_bool(cfg.get("share_query")),
        fusion_mode=cfg.get("fusion_mode", "static"),
        vision_layer_indices=_as_int_list(cfg.get("vision_layer_indices"), [9, 17, 24]),
        router_dim=int(cfg.get("router_dim", 256)),
        use_moe=_as_bool(cfg.get("use_moe")),
        num_experts=int(cfg.get("num_experts", 4)),
        moe_top_k=int(cfg.get("moe_top_k", 2)),
        load_balance_coef=float(cfg.get("load_balance_coef", 0.01)),
        use_merger_adapter=_as_bool(cfg.get("use_merger_adapter")),
        merger_adapter_rank=int(cfg.get("merger_adapter_rank", 64)),
        use_nested_guided_fusion=_as_bool(cfg.get("use_nested_guided_fusion")),
        ngf_layer_weight_mode=cfg.get("ngf_layer_weight_mode", "text"),
        ngf_inner_guiding=_as_bool(cfg.get("ngf_inner_guiding"), default=True),
        nested_replace_base=_as_bool(cfg.get("nested_replace_base"), default=True),
        use_ngf_sequential=_as_bool(cfg.get("use_ngf_sequential"), default=False),
        layer_balance_coef=float(cfg.get("layer_balance_coef", 0.0)),
        ngf_tap=cfg.get("ngf_tap", "block"),
        ngf_intermediate_only=_as_bool(cfg.get("ngf_intermediate_only"), default=False),
        ngf_full_connector=_as_bool(cfg.get("ngf_full_connector"), default=False),
        use_post_merger_alf=_as_bool(cfg.get("use_post_merger_alf"), default=False),
        post_merger_adapter=_as_bool(cfg.get("post_merger_adapter"), default=True),
        post_merger_adapter_rank=int(cfg.get("post_merger_adapter_rank", 64)),
        alf_router_dim=int(cfg.get("alf_router_dim", 256)),
    )
    if args.vlm_model_id is not None:
        model_kwargs["model_id"] = args.vlm_model_id
    elif cfg.get("vlm_model_id") not in (None, "None", ""):
        model_kwargs["model_id"] = cfg["vlm_model_id"]
    if args.revision is not None:
        model_kwargs["revision"] = args.revision

    print("Building model and loading components...")
    model = OV2RoutedMaTCA(**model_kwargs)
    model.load_classifier(checkpoint_path, strict=False)
    model.eval()

    # Accumulators.
    all_labels, all_probs, all_preds = [], [], []
    expert_usage_acc = None
    gate_entropy_acc, gate_entropy_n = 0.0, 0
    layer_weight_acc = None
    depth_weight_acc = None
    vision_fusion_weight_acc = None
    ngf_gate_entropy_acc, ngf_gate_entropy_n = 0.0, 0
    flips, prob_shift, grounding_n = 0, 0.0, 0

    alt_task = "a completely unrelated background scene with no robot activity"

    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        entries = dataset[start:end]
        tasks = entries["task"]
        images = entries["images"]
        prompt_styles = entries.get("prompt_style", [None] * len(tasks))

        if args.task_substitution is not None:
            tasks = [args.task_substitution] * len(tasks)

        labels = [float(label_to_binary(label)) for label in entries["label"]]

        try:
            preds, probs = model.predict(
                images, tasks, prompt_styles=prompt_styles,
                prediction_mode=args.prediction_mode,
            )
        except Exception as exc:
            print(f"[warn] skipping batch at {start}: {exc}")
            continue

        all_labels.extend(labels)
        all_probs.extend(probs.detach().float().cpu().tolist())
        all_preds.extend(preds.detach().float().cpu().tolist())

        # MoE / layer diagnostics from the last forward.
        if model.moe is not None and model.moe.last_expert_usage is not None:
            usage = model.moe.last_expert_usage.detach().float().cpu().numpy()
            expert_usage_acc = usage if expert_usage_acc is None else expert_usage_acc + usage
            if model.moe.last_gate_logits is not None:
                gate_entropy_acc += gate_entropy_from_logits(model.moe.last_gate_logits)
                gate_entropy_n += 1
        if model._last_layer_weights is not None:
            lw = model._last_layer_weights.detach().float().cpu().numpy().reshape(-1)
            layer_weight_acc = lw if layer_weight_acc is None else layer_weight_acc + lw
        if model._last_depth_weights is not None:
            dw = model._last_depth_weights.detach().float().cpu().numpy().reshape(-1)
            depth_weight_acc = dw if depth_weight_acc is None else depth_weight_acc + dw
        if model._last_vision_fusion_weights is not None:
            vw = model._last_vision_fusion_weights.detach().float().cpu().numpy().reshape(-1)
            vision_fusion_weight_acc = (
                vw if vision_fusion_weight_acc is None else vision_fusion_weight_acc + vw
            )
        if model.nested_fusion is not None and model.nested_fusion.last_gate_entropy is not None:
            ngf_gate_entropy_acc += float(model.nested_fusion.last_gate_entropy)
            ngf_gate_entropy_n += 1

        # Grounding-leakage probe: swap the query text on the SAME images.
        if args.grounding_check:
            try:
                preds_alt, probs_alt = model.predict(
                    images, [alt_task] * len(tasks), prompt_styles=prompt_styles,
                    prediction_mode=args.prediction_mode,
                )
                flips += int((preds_alt != preds).sum().item())
                prob_shift += float((probs_alt - probs).abs().sum().item())
                grounding_n += len(tasks)
            except Exception as exc:
                print(f"[warn] grounding probe failed at {start}: {exc}")

    labels_np = np.asarray(all_labels)
    probs_np = np.asarray(all_probs)
    preds_np = np.asarray(all_preds)

    results = {
        "checkpoint": checkpoint_path,
        "dataset": args.dataset_name,
        "split": args.split,
        "pov": args.pov,
        "num_samples": int(len(labels_np)),
        "prediction_mode": args.prediction_mode,
        "config": {k: cfg.get(k) for k in (
            "use_hier_fusion", "use_tgif_fusion", "use_router", "use_fuse_then_route", "use_hier_router",
            "depth_fusion_mode", "use_moe", "router_mode", "gate_type",
            "gate_style", "num_experts", "moe_top_k", "load_balance_coef", "fusion_mode",
            "use_merger_adapter", "merger_adapter_rank", "pooling_mode",
            "use_nested_guided_fusion", "ngf_layer_weight_mode", "ngf_inner_guiding",
            "nested_replace_base", "use_ngf_sequential", "layer_balance_coef",
            "use_post_merger_alf", "post_merger_adapter", "post_merger_adapter_rank",
            "alf_router_dim",
        )},
    }

    if len(labels_np) > 0:
        results["accuracy"] = float(accuracy_score(labels_np, preds_np))
        results["precision"] = float(precision_score(labels_np, preds_np, zero_division=0))
        results["recall"] = float(recall_score(labels_np, preds_np, zero_division=0))
        results["f1"] = float(f1_score(labels_np, preds_np, zero_division=0))
        results["confusion_matrix"] = confusion_matrix(labels_np, preds_np).tolist()
        results["brier"] = float(np.mean((probs_np - labels_np) ** 2))
        results["ece"] = expected_calibration_error(probs_np, labels_np)
        if len(np.unique(labels_np)) > 1:
            results["auroc"] = float(roc_auc_score(labels_np, probs_np))
            results["average_precision"] = float(average_precision_score(labels_np, probs_np))

    if expert_usage_acc is not None:
        denom = expert_usage_acc.sum() or 1.0
        results["expert_usage_hist"] = (expert_usage_acc / denom).tolist()
        if gate_entropy_n:
            results["mean_gate_entropy_nats"] = gate_entropy_acc / gate_entropy_n
    if layer_weight_acc is not None:
        denom = layer_weight_acc.sum() or 1.0
        results["layer_fusion_weights"] = (layer_weight_acc / denom).tolist()
    if depth_weight_acc is not None:
        denom = depth_weight_acc.sum() or 1.0
        results["depth_weights"] = (depth_weight_acc / denom).tolist()
    if vision_fusion_weight_acc is not None:
        denom = vision_fusion_weight_acc.sum() or 1.0
        results["vision_fusion_weights"] = (vision_fusion_weight_acc / denom).tolist()
        if _as_bool(cfg.get("use_nested_guided_fusion")):
            results["ngf_layer_alpha"] = results["vision_fusion_weights"]
    if ngf_gate_entropy_n:
        results["ngf_mean_patch_gate_entropy"] = ngf_gate_entropy_acc / ngf_gate_entropy_n
    nested_fusion = getattr(model, "nested_fusion", None)
    if nested_fusion is not None:
        inner_betas = getattr(nested_fusion, "inner_betas", None)
        if inner_betas is not None:
            results["ngf_inner_beta"] = inner_betas.detach().float().cpu().tolist()
        beta_seq = getattr(nested_fusion, "beta_seq", None)
        if beta_seq is not None:
            results["ngf_seq_beta"] = beta_seq.detach().float().cpu().tolist()
        last_seq_gate = getattr(nested_fusion, "last_seq_gate", None)
        if last_seq_gate is not None:
            results["ngf_seq_mean_gate"] = last_seq_gate.detach().float().cpu().tolist()
    post_merger_fusion = getattr(model, "post_merger_fusion", None)
    if post_merger_fusion is not None:
        results["alf_beta"] = float(post_merger_fusion.beta.detach().item())
        if post_merger_fusion.last_alpha is not None:
            results["alf_depth_attn"] = post_merger_fusion.last_alpha.detach().float().cpu().tolist()
    post_merger_adapter = getattr(model, "post_merger_adapter", None)
    if post_merger_adapter is not None:
        results["alf_adapter_gamma"] = float(post_merger_adapter.gamma.detach().item())
    if args.grounding_check and grounding_n:
        results["grounding_flip_rate"] = flips / grounding_n
        results["grounding_mean_prob_shift"] = prob_shift / grounding_n
    if args.task_substitution is not None:
        results["task_substitution"] = args.task_substitution

    out_path = os.path.join(args.result_folder, "results.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
