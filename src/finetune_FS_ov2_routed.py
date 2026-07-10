"""Training driver for the OneVision-2 routed MaTCA model (with optional MoE).

Trains the post-LLM MaTCA head and the optional pre-LLM Stage-1 modules
(hierarchical vision fusion, dual-query router, Mixture-of-Failure-Experts) on
top of a frozen ``LLaVA-OneVision-2`` backbone for binary robot success/failure
detection.

Datasets are loaded with the shared ``load_data`` helper (CALVIN / DROID / AHA).
CALVIN/DROID provide binary labels only (no failure-mode annotations), so the
MoE experts specialize latently; the optional ``--moe_gate_supervision`` flag is
a no-op unless a ``failure_mode_id`` column is present.

Routed runs (router / hier fusion / MoE) backprop through the frozen 8B LM and
therefore force ``batch_size=1`` regardless of ``--batch_size`` (gradient
checkpointing is enabled in the model).
"""

import argparse
import os
import random

import numpy as np
import torch

from load_dataset import augment_droid_dataset, load_data
from model_ov2_routed_matca import OV2RoutedMaTCA, train_model


def set_global_seed(seed):
    """Seed python / numpy / torch (CPU + CUDA) for multi-seed reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] global seed set to {seed}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune OneVision-2 routed MaTCA for robot failure detection."
    )

    # ----- Backbone / data -----
    parser.add_argument("--vlm_model_id", type=str, default=None,
                        help="Path or HF id of the OneVision-2 checkpoint (default: model default).")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional checkpoint revision.")
    parser.add_argument("--dataset_name", type=str, default="calvin",
                        choices=["calvin", "droid", "aha"])
    parser.add_argument("--pov", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--style", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--num_entry", type=str, default="full",
                        help="'full' or an integer number of training samples.")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable synthetic class balancing for DROID "
                             "(by default DROID train/val are balanced via "
                             "augment_droid_dataset, matching the Qwen pipeline).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels", type=int, default=200704)

    # ----- Head configuration -----
    parser.add_argument("--num_classifiers", type=int, default=3)
    parser.add_argument("--target_layer_indices", type=int, nargs="+", default=[19, 28, 36],
                        help="LM hidden-state indices used by the post-LLM heads.")
    parser.add_argument("--pooling_mode", type=str, default="tcond",
                        choices=["tcond", "hybrid", "last_token", "text_mean"])
    parser.add_argument("--fusion_mode", type=str, default="static",
                        choices=["static", "dynamic", "mean", "text"],
                        help="ViT depth mixing for hier/fuse-then-route: static, dynamic "
                             "(visual pool), mean, or text (TGIF-style task+fail router).")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # ----- Stage-1 (pre-LLM) toggles -----
    parser.add_argument("--use_hier_fusion", action="store_true",
                        help="Enable hierarchical vision-layer fusion before the merger.")
    parser.add_argument("--use_tgif_fusion", action="store_true",
                        help="TGIF-style complete multi-depth feature mixture (layers "
                             "9/17/24 + base) with no router; pairs with --use_merger_adapter.")
    parser.add_argument("--use_fuse_then_route", action="store_true",
                        help="Fuse multi-depth vision features (TGIF-style), then apply one "
                             "flat dual-query router on the fused tokens. Equivalent to "
                             "--use_hier_fusion --use_router but avoids per-depth delta "
                             "softmax collapse from --use_hier_router.")
    parser.add_argument("--vision_layer_indices", type=int, nargs="*", default=[9, 17, 24],
                        help="Encoder hidden_states indices for hier fusion/router "
                             "(0=post-embed; for 24-block OV2 ViT, valid 0..24). Pass with no "
                             "values for NGF base-only ablation (L=1).")
    parser.add_argument("--use_router", action="store_true",
                        help="Enable the dual task/failure query router.")
    parser.add_argument("--use_hier_router", action="store_true",
                        help="Enable hierarchical per-depth dual-query router "
                             "(mutually exclusive with --use_router, --use_hier_fusion, --use_moe).")
    parser.add_argument("--depth_fusion_mode", type=str, default="query_cond",
                        choices=["query_cond", "static"],
                        help="Depth weighting for --use_hier_router (query-conditioned or static).")
    parser.add_argument("--router_mode", type=str, default="contrastive",
                        choices=["task_only", "task_fail", "contrastive"])
    parser.add_argument("--gate_type", type=str, default="sigmoid",
                        choices=["sigmoid", "softmax"])
    parser.add_argument("--gate_style", type=str, default="multiplicative",
                        choices=["multiplicative", "guiding"],
                        help="QMSA ablation: multiplicative gates vs additive-bias guiding.")
    parser.add_argument("--share_query", action="store_true",
                        help="Share the task/failure query projection.")
    parser.add_argument("--router_dim", type=int, default=256)

    # ----- Mixture-of-Failure-Experts -----
    parser.add_argument("--use_moe", action="store_true",
                        help="Replace the router's phi with a Mixture-of-Failure-Experts.")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--load_balance_coef", type=float, default=0.01,
                        help="Relaxed load-balance weight (small/0 lets rare experts specialize).")
    parser.add_argument("--moe_gate_supervision", action="store_true",
                        help="Use failure_mode_id (if present) to supervise the expert gate.")
    parser.add_argument("--use_merger_adapter", action="store_true",
                        help="Train a parallel low-rank adapter on the frozen patch merger.")
    parser.add_argument("--merger_adapter_rank", type=int, default=64,
                        help="Bottleneck rank for --use_merger_adapter.")

    # ----- Nested Guided Fusion (NGF) -----
    parser.add_argument("--use_nested_guided_fusion", action="store_true",
                        help="Per-(layer,patch) dual-query guiding gates + TGIF-style task "
                             "layer weights; fused token field replaces V_base. Mutually "
                             "exclusive with all other Stage-1 toggles.")
    parser.add_argument("--ngf_layer_weight_mode", type=str, default="text",
                        choices=["text", "contrastive", "static", "uniform", "token"],
                        help="Outer depth weights alpha_l: task-text routed (text), "
                             "task−fail contrastive over depth summaries (contrastive; "
                             "M6), static learnable, uniform (1/L), or per-patch "
                             "alpha_{l,n} (token; Arch B).")
    parser.add_argument("--ngf_sequential", action="store_true",
                        help="Arch C: depth-recurrent residual refinement "
                             "(SequentialNestedFusion) instead of parallel weighted fuse. "
                             "Overrides --ngf_layer_weight_mode (no outer alpha).")
    parser.add_argument("--ngf_no_inner", action="store_true",
                        help="Disable inner per-patch guiding (h_l = raw V_l); TGIF-style "
                             "outer fusion only (ablation A2).")
    parser.add_argument("--nested_residual", action="store_true",
                        help="Blend fused field as V_base + eta*(F - V_base), eta init 0, "
                             "instead of fully replacing V_base (ablation A5).")
    parser.add_argument("--layer_balance_coef", type=float, default=0.0,
                        help="Entropy load-balance on NGF depth weights alpha_l, "
                             "category-aggregator alpha, and post-merger ALF depth attention.")
    parser.add_argument("--ngf_inner_beta_l2", type=float, default=0.0,
                        help="L2 shrinkage on NGF inner-residual scales (beta_l / beta_seq). "
                             "Dials back inner-guiding capacity toward inner-OFF for better "
                             "cross-dataset transfer (e.g. 0.01).")
    parser.add_argument("--ngf_tap", type=str, default="block",
                        choices=["block", "ffn_act"],
                        help="NGF layer-token source: 'block' (encoder hidden_states, "
                             "vision_layer_indices index the hidden_states tuple) or "
                             "'ffn_act' (post-GELU FFN activation; vision_layer_indices "
                             "are treated as 0-based encoder BLOCK indices).")
    parser.add_argument("--ngf_intermediate_only", action="store_true",
                        help="Fuse intermediate layers only: do NOT add V_base as the "
                             "last NGF depth (honest intermediate-layer transfer test).")
    parser.add_argument("--ngf_full_connector", action="store_true",
                        help="Replace the frozen patch merger with a fully-trainable, "
                             "warm-started clone (NGF or category-agg path; mutually "
                             "exclusive with --use_merger_adapter).")

    # ----- Post-merger ALF fusion (fuse-after-merger) -----
    parser.add_argument("--use_post_merger_alf", action="store_true",
                        help="Fuse-after-merger: pass each intermediate ViT depth "
                             "through the frozen merger separately, then ALF-style "
                             "cross-attention in 4096-d LLM space with H_base anchor. "
                             "Mutually exclusive with all other Stage-1 toggles and "
                             "--use_merger_adapter (use --post_merger_adapter instead). "
                             "Pass intermediate-only --vision_layer_indices (exclude 24).")
    parser.add_argument("--no_post_merger_adapter", action="store_true",
                        help="Disable the 4096-d post-merger output adapter "
                             "(on by default with --use_post_merger_alf).")
    parser.add_argument("--post_merger_adapter_rank", type=int, default=64,
                        help="Bottleneck rank for the post-merger output adapter.")
    parser.add_argument("--alf_router_dim", type=int, default=256,
                        help="Q/K projection dim for the post-merger ALF cross-attention.")

    # ----- Category contrastive aggregator (IGVA-style 6×4) -----
    parser.add_argument("--use_category_aggregator", action="store_true",
                        help="6-category ViT aggregator: mean-pool 4 layers per category, "
                             "contrastive task−fail weights over categories, concat with "
                             "V_base and low-rank adapter (gamma init 0). Mutually exclusive "
                             "with NGF / router / post-merger ALF.")
    parser.add_argument("--category_adapter_rank", type=int, default=256,
                        help="Bottleneck rank for concat adapter [V_base; F] -> delta.")
    parser.add_argument("--category_concat_mode", type=str, default="residual_last",
                        choices=["residual_last", "igva_penultimate", "igva_base"],
                        help="residual_last: [V_base;F]+γ residual (default). "
                             "igva_penultimate: paper-style [F;F_pen] -> adapter. "
                             "igva_base: patch-VLM [F;V_base] -> adapter.")
    parser.add_argument("--category_penultimate_index", type=int, default=23,
                        help="hidden_states index for penultimate layer in igva_penultimate mode.")

    # ----- Optimization -----
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Forced to 1 when any Stage-1 module is enabled.")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage1_lr", type=float, default=None,
                        help="Optional higher LR for Stage-1 modules (router, "
                             "fusion, merger adapters). MaTCA head uses --lr.")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--loss_mode", type=str, default="fusion",
                        choices=["fusion", "per_head", "all"])
    parser.add_argument("--prediction_mode", type=str, default="fusion",
                        choices=["fusion", "head_average", "head_majority"])

    # ----- IO -----
    parser.add_argument("--result_folder", type=str, default="./results_ov2_routed")
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main():
    args = build_parser().parse_args()

    # Arch C (sequential) has no outer alpha; it overrides the depth-weight mode.
    if args.ngf_sequential and args.ngf_layer_weight_mode != "text":
        print(
            "[warn] --ngf_sequential ignores --ngf_layer_weight_mode "
            f"('{args.ngf_layer_weight_mode}'): sequential refinement has no outer alpha."
        )

    set_global_seed(args.seed)

    num_entry = args.num_entry
    if num_entry != "full":
        num_entry = int(num_entry)

    os.makedirs(args.result_folder, exist_ok=True)

    print("Loading training split...")
    train_dataset = load_data(
        dataset_name=args.dataset_name,
        style=args.style,
        pov=args.pov,
        split="train",
        num_entry=num_entry,
        seed=args.seed,
    )
    print("Loading validation/test split...")
    val_dataset = load_data(
        dataset_name=args.dataset_name,
        style=args.style,
        pov=args.pov,
        split="test",
        num_entry=num_entry if num_entry == "full" else min(num_entry, 200),
        seed=args.seed,
    )

    # DROID HF splits are single-class (all success); synthesize a balanced
    # failure half via task-shift augmentation, matching the Qwen pipeline.
    if args.dataset_name == "droid" and not args.no_augmentation:
        print(f"Augmenting DROID train split ({len(train_dataset)} -> balanced)...")
        train_dataset = augment_droid_dataset(train_dataset)
        print(f"  train now {len(train_dataset)} samples")
        print(f"Augmenting DROID validation split ({len(val_dataset)} -> balanced)...")
        val_dataset = augment_droid_dataset(val_dataset)
        print(f"  validation now {len(val_dataset)} samples")

    model_kwargs = dict(
        device=args.device,
        max_pixels=args.max_pixels,
        num_classifiers=args.num_classifiers,
        target_layer_indices=args.target_layer_indices,
        pooling_mode=args.pooling_mode,
        dropout_rate=args.dropout_rate,
        use_hier_fusion=args.use_hier_fusion,
        use_tgif_fusion=args.use_tgif_fusion,
        use_fuse_then_route=args.use_fuse_then_route,
        use_router=args.use_router,
        use_hier_router=args.use_hier_router,
        depth_fusion_mode=args.depth_fusion_mode,
        router_mode=args.router_mode,
        gate_type=args.gate_type,
        gate_style=args.gate_style,
        share_query=args.share_query,
        fusion_mode=args.fusion_mode,
        vision_layer_indices=args.vision_layer_indices,
        router_dim=args.router_dim,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        load_balance_coef=args.load_balance_coef,
        moe_gate_supervision=args.moe_gate_supervision,
        use_merger_adapter=args.use_merger_adapter,
        merger_adapter_rank=args.merger_adapter_rank,
        use_nested_guided_fusion=args.use_nested_guided_fusion,
        ngf_layer_weight_mode=args.ngf_layer_weight_mode,
        ngf_inner_guiding=not args.ngf_no_inner,
        nested_replace_base=not args.nested_residual,
        use_ngf_sequential=args.ngf_sequential,
        ngf_inner_beta_l2=args.ngf_inner_beta_l2,
        layer_balance_coef=args.layer_balance_coef,
        ngf_tap=args.ngf_tap,
        ngf_intermediate_only=args.ngf_intermediate_only,
        ngf_full_connector=args.ngf_full_connector,
        use_post_merger_alf=args.use_post_merger_alf,
        post_merger_adapter=not args.no_post_merger_adapter,
        post_merger_adapter_rank=args.post_merger_adapter_rank,
        alf_router_dim=args.alf_router_dim,
        use_category_aggregator=args.use_category_aggregator,
        category_adapter_rank=args.category_adapter_rank,
        category_concat_mode=args.category_concat_mode,
        category_penultimate_index=args.category_penultimate_index,
    )
    if args.vlm_model_id is not None:
        model_kwargs["model_id"] = args.vlm_model_id
    if args.revision is not None:
        model_kwargs["revision"] = args.revision

    print("Building OV2RoutedMaTCA model...")
    model = OV2RoutedMaTCA(**model_kwargs)

    config = {
        "lr": args.lr,
        "stage1_lr": args.stage1_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "loss_mode": args.loss_mode,
        "prediction_mode": args.prediction_mode,
        "save_path": args.result_folder,
    }

    # Persist the run configuration for reproducible evaluation.
    config_path = os.path.join(args.result_folder, "config.txt")
    with open(config_path, "w") as handle:
        for key, value in vars(args).items():
            handle.write(f"{key}={value}\n")
    print(f"Wrote run config to {config_path}")

    best_val = train_model(model, train_dataset, val_dataset, config)
    print(f"Best validation accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()
