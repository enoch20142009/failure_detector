"""Phase 0 / 0b - per-(layer,module) linear probes of the OV2 vision tower.

Two probe modes (``--probe_mode``):

**vision_only** (Phase 0):
  Frozen ViT, mean-pool patch tokens, logistic regression. Tests raw visual
  failure separability without task text.

**text_guided** (Phase 0b):
  Same frozen ViT taps, but features are built with frozen task/failure query
  embeddings (LM embed table, no LM forward) and fixed random key projections
  that task-condition *which patches are pooled* - mirroring Stage-1 routing
  geometry without training W_k/W_q/phi. Then logistic regression isolates
  which depth/module is easiest to read out success vs fail *with text guidance*.

Also reports domain-invariance (CALVIN vs DROID/AHA) and, for text_guided,
grounding sensitivity on CALVIN test (score change when task query is swapped,
images held fixed).

Example:
    python src/probe_vit_modules.py \\
        --probe_mode text_guided \\
        --vlm_model_id /scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct \\
        --pov 1 --num_entry 800 \\
        --result_folder ./eval_results/ov2_probe_phase0b
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from load_dataset import load_data
from model_ov2_baseline import (
    DEFAULT_OV2_MODEL_ID,
    DEFAULT_OV2_REVISION,
    build_messages,
    label_to_binary,
)

MODULES = ("Act", "LN2", "RC2")
HIGHLIGHT_LAYERS = (6, 12, 18)
FAIL_TEMPLATE = (
    "Visual evidence that the following robot task was not successfully "
    "completed: {task}"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Per-(layer,module) linear probe of the OV2 vision tower."
    )
    parser.add_argument(
        "--probe_mode",
        type=str,
        default="both",
        choices=["vision_only", "text_guided", "both"],
        help="vision_only=mean-pool (Phase 0); text_guided=query-guided pool "
             "(Phase 0b); both=run and report both.",
    )
    parser.add_argument(
        "--text_pool_mode",
        type=str,
        default="concat",
        choices=["task", "fail", "contrast", "concat"],
        help="How to combine task/fail guided pools into the probe feature "
             "(text_guided mode). 'concat' = [pool_task; pool_fail; contrast].",
    )
    parser.add_argument("--vlm_model_id", type=str, default=DEFAULT_OV2_MODEL_ID)
    parser.add_argument("--revision", type=str, default=DEFAULT_OV2_REVISION)
    parser.add_argument("--pov", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--style", type=str, default="image", choices=["image", "video"])
    parser.add_argument(
        "--num_entry",
        type=str,
        default="800",
        help="'full' or integer cap for CALVIN train (test/domain splits use full).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Encoder layer indices (default: all 0..N-1).",
    )
    parser.add_argument("--key_dim", type=int, default=256,
                        help="Fixed projection width for text-guided gating.")
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument(
        "--probe_C",
        type=float,
        default=1.0,
        help="Inverse L2 for logistic regression (smaller = stronger reg).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--result_folder",
        type=str,
        default="./eval_results/ov2_probe_phase0",
    )
    parser.add_argument(
        "--merge_existing_results",
        action="store_true",
        help="When re-running one probe_mode, merge into existing probe_results.json "
             "instead of overwriting other modes.",
    )
    return parser


def _parse_num_entry(num_entry):
    if num_entry in (None, "full"):
        return "full"
    return int(num_entry)


class FixedTextGuidedProjector:
    """Fixed random projections for query->key gating (not trained)."""

    def __init__(self, module_dims, query_dim, key_dim, seed):
        rng = np.random.RandomState(seed)
        self.key_dim = key_dim
        self.W_k = {}
        for kind, dim in module_dims.items():
            w = rng.randn(dim, key_dim).astype(np.float64)
            w /= np.linalg.norm(w, axis=0, keepdims=True).clip(min=1e-8)
            self.W_k[kind] = w
        wq = rng.randn(query_dim, key_dim).astype(np.float64)
        wq /= np.linalg.norm(wq, axis=0, keepdims=True).clip(min=1e-8)
        self.W_q = wq

    def guided_pool(self, v_patches, q_task, q_fail, kind, pool_mode):
        """Task/failure softmax pooling over patches -> feature vector."""
        v = np.asarray(v_patches, dtype=np.float64)
        k = v @ self.W_k[kind]
        qt = (np.asarray(q_task, dtype=np.float64).reshape(-1) @ self.W_q).reshape(1, -1)
        qf = (np.asarray(q_fail, dtype=np.float64).reshape(-1) @ self.W_q).reshape(1, -1)
        scale = np.sqrt(self.key_dim)

        wt = self._softmax(k @ qt.T / scale, axis=0)
        wf = self._softmax(k @ qf.T / scale, axis=0)
        pool_t = (wt * v).sum(axis=0)
        pool_f = (wf * v).sum(axis=0)
        contrast = pool_t - pool_f

        if pool_mode == "task":
            return pool_t
        if pool_mode == "fail":
            return pool_f
        if pool_mode == "contrast":
            return contrast
        return np.concatenate([pool_t, pool_f, contrast], axis=0)

    @staticmethod
    def _softmax(x, axis=0):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(np.clip(x, -30, 30))
        return e / e.sum(axis=axis, keepdims=True)


class VisionFeatureExtractor:
    """Frozen OV2 vision tower with Act / LN2 / RC2 hooks per layer."""

    def __init__(self, model_id, revision, device, max_pixels, layers=None):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        if max_pixels is not None:
            self.processor.image_processor.max_pixels = int(max_pixels)
            if hasattr(self.processor.image_processor, "size"):
                self.processor.image_processor.size["longest_edge"] = int(max_pixels)

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, dtype=torch.bfloat16
        )
        self.vlm.to(self.device)
        self.vlm.eval()
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.visual = self.vlm.model.visual
        self.encoder_layers = self.visual.encoder.layers
        self.num_layers = len(self.encoder_layers)
        if layers is None:
            self.layers = list(range(self.num_layers))
        else:
            for idx in layers:
                if idx < 0 or idx >= self.num_layers:
                    raise ValueError(
                        f"layer {idx} out of range 0..{self.num_layers - 1}"
                    )
            self.layers = list(layers)

        self.query_dim = self.vlm.config.text_config.hidden_size
        self.module_dims = {
            "Act": self.visual.config.intermediate_size,
            "LN2": self.visual.config.hidden_size,
            "RC2": self.visual.config.hidden_size,
        }

        self._cache = {}
        self._handles = []
        self._register_hooks(store_patches=True)

    def _register_hooks(self, store_patches=True):
        def make_hook(layer_idx, kind):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                t = tensor.detach().float()
                if store_patches:
                    # [1, num_patches, dim] -> [num_patches, dim]
                    self._cache[(kind, layer_idx)] = t.squeeze(0).cpu().numpy()
                else:
                    self._cache[(kind, layer_idx)] = t.mean(dim=1).squeeze(0).cpu().numpy()

            return hook

        for idx in self.layers:
            layer = self.encoder_layers[idx]
            self._handles.append(
                layer.mlp.activation_fn.register_forward_hook(make_hook(idx, "Act"))
            )
            self._handles.append(
                layer.layer_norm2.register_forward_hook(make_hook(idx, "LN2"))
            )
            self._handles.append(
                layer.register_forward_hook(make_hook(idx, "RC2"))
            )

    def _prepare_vision_inputs(self, images):
        if not isinstance(images, list):
            images = [images]
        messages = build_messages(images=images, task="probe", prompt_style=None)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt", padding=True
        )
        prepared = {}
        for key, value in inputs.items():
            prepared[key] = value.to(self.device) if torch.is_tensor(value) else value
        return prepared

    @torch.inference_mode()
    def compute_queries(self, task):
        """Frozen LM embedding-table queries (same strings as Stage 1 router)."""
        embed = self.vlm.get_input_embeddings()
        tokenizer = self.processor.tokenizer

        def pooled(text):
            ids = tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.device)
            emb = embed(ids).float().mean(dim=1).squeeze(0)
            return emb.cpu().numpy()

        q_task = pooled(task)
        q_fail = pooled(FAIL_TEMPLATE.format(task=task))
        return q_task, q_fail

    @torch.inference_mode()
    def extract_patches(self, images):
        """Vision forward only; returns patch tokens per (module, layer)."""
        inputs = self._prepare_vision_inputs(images)
        pe_dtype = self.visual.embeddings.patch_embedding.weight.dtype
        pixel_values = inputs["pixel_values"].type(pe_dtype)
        self._cache.clear()
        self.visual(
            pixel_values,
            grid_thw=inputs.get("image_grid_thw"),
            patch_positions=inputs.get("patch_positions"),
        )
        return {
            kind: {idx: self._cache[(kind, idx)] for idx in self.layers}
            for kind in MODULES
        }

    @torch.inference_mode()
    def extract_mean_pooled(self, images):
        """Legacy mean-pool path for vision_only mode."""
        patches = self.extract_patches(images)
        return {
            kind: {idx: patches[kind][idx].mean(axis=0) for idx in self.layers}
            for kind in MODULES
        }

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _feats_from_patches(patches, q_task, q_fail, projector, pool_mode, layers):
    """Build [L, D_feat] guided features for one sample."""
    out = {}
    for kind in MODULES:
        vecs = []
        for idx in layers:
            vec = projector.guided_pool(
                patches[kind][idx], q_task, q_fail, kind, pool_mode
            )
            vecs.append(vec)
        out[kind] = np.stack(vecs, axis=0)
    return out


def extract_split(extractor, dataset, split_name, mode, projector=None, pool_mode="concat"):
    """Extract pooled probe features for one dataset split."""
    n = len(dataset)
    feats = {kind: [] for kind in MODULES}
    labels = []
    tasks = []

    for i in range(n):
        sample = dataset[i]
        tasks.append(sample["task"])
        labels.append(label_to_binary(sample["label"]))

        if mode == "vision_only":
            result = extractor.extract_mean_pooled(sample["images"])
            for kind in MODULES:
                feats[kind].append(
                    np.stack([result[kind][idx] for idx in extractor.layers], axis=0)
                )
        else:
            patches = extractor.extract_patches(sample["images"])
            q_task, q_fail = extractor.compute_queries(sample["task"])
            guided = _feats_from_patches(
                patches, q_task, q_fail, projector, pool_mode, extractor.layers
            )
            for kind in MODULES:
                feats[kind].append(guided[kind])

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{split_name}] {i + 1}/{n} samples", flush=True)

    feats = {kind: np.stack(arrs, axis=0) for kind, arrs in feats.items()}
    return feats, np.asarray(labels, dtype=np.int64), tasks


def roc_auc_score(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    rank = 1
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        rank += (j - i + 1)
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fit_logreg(x_train, y_train, max_iter, C=1.0):
    xt = torch.from_numpy(np.ascontiguousarray(x_train)).double()
    yt = torch.from_numpy(np.ascontiguousarray(y_train)).double()
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    reg = 1.0 / max(C, 1e-8)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = xt @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, yt, reduction="sum")
        loss = loss + 0.5 * reg * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach().numpy()[0])


def _predict_proba(x, w, b, mean, std):
    x = (x.astype(np.float64) - mean) / std
    logits = np.clip(x @ w + b, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _probe_pair(feats_train, y_train, feats_eval, y_eval, layers, max_iter, probe_c):
    results = {}
    models = {}
    for kind in MODULES:
        results[kind] = {}
        models[kind] = {}
        for li, layer_idx in enumerate(layers):
            x_tr = feats_train[kind][:, li, :].astype(np.float64)
            mean = x_tr.mean(axis=0)
            std = x_tr.std(axis=0)
            std[std == 0] = 1.0

            if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
                results[kind][layer_idx] = float("nan")
                continue
            w, b = _fit_logreg((x_tr - mean) / std, y_train, max_iter, C=probe_c)
            x_ev = feats_eval[kind][:, li, :]
            scores = _predict_proba(x_ev, w, b, mean, std)
            results[kind][layer_idx] = roc_auc_score(y_eval, scores)
            models[kind][layer_idx] = (w, b, mean, std)
    return results, models


def probe_grounding_sensitivity(
    dataset, task_list, models, layers, projector, pool_mode, extractor, seed
):
    """Score shift when task/fail queries are swapped to another sample's task.

    Processes one image at a time so full patch tensors are never retained for
    the whole split (avoids multi‑tens‑of‑GB RAM use on CALVIN test).
    """
    n = len(dataset)
    rng = np.random.RandomState(seed)
    wrong_idx = rng.randint(0, n, size=n)

    sums = {kind: {layer_idx: 0.0 for layer_idx in layers} for kind in MODULES}
    counts = {kind: {layer_idx: 0 for layer_idx in layers} for kind in MODULES}

    for i in range(n):
        sample = dataset[i]
        patches = extractor.extract_patches(sample["images"])
        q_task, q_fail = extractor.compute_queries(sample["task"])
        j = int(wrong_idx[i])
        if j == i:
            j = (i + 1) % n
        q_task_w, q_fail_w = extractor.compute_queries(task_list[j])

        for kind in MODULES:
            for layer_idx in layers:
                if layer_idx not in models.get(kind, {}):
                    continue
                w, b, mean, std = models[kind][layer_idx]
                feat_ok = projector.guided_pool(
                    patches[kind][layer_idx], q_task, q_fail, kind, pool_mode
                )
                feat_bad = projector.guided_pool(
                    patches[kind][layer_idx], q_task_w, q_fail_w, kind, pool_mode
                )
                p_ok = _predict_proba(feat_ok.reshape(1, -1), w, b, mean, std)[0]
                p_bad = _predict_proba(feat_bad.reshape(1, -1), w, b, mean, std)[0]
                sums[kind][layer_idx] += abs(p_ok - p_bad)
                counts[kind][layer_idx] += 1

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [grounding] {i + 1}/{n} samples", flush=True)

    out = {kind: {} for kind in MODULES}
    for kind in MODULES:
        for layer_idx in layers:
            if counts[kind][layer_idx] == 0:
                out[kind][layer_idx] = float("nan")
            else:
                out[kind][layer_idx] = sums[kind][layer_idx] / counts[kind][layer_idx]
    return out


def probe_domain(feats_a, feats_b, layers, max_iter, probe_c, seed):
    rng = np.random.RandomState(seed)
    na = feats_a[MODULES[0]].shape[0]
    nb = feats_b[MODULES[0]].shape[0]
    m = min(na, nb)
    idx_a = rng.permutation(na)[:m]
    idx_b = rng.permutation(nb)[:m]
    n_tr = int(round(m * 0.7))
    a_tr, a_ev = idx_a[:n_tr], idx_a[n_tr:]
    b_tr, b_ev = idx_b[:n_tr], idx_b[n_tr:]

    feats_train, feats_eval = {}, {}
    for kind in MODULES:
        feats_train[kind] = np.concatenate(
            [feats_a[kind][a_tr], feats_b[kind][b_tr]], axis=0
        )
        feats_eval[kind] = np.concatenate(
            [feats_a[kind][a_ev], feats_b[kind][b_ev]], axis=0
        )
    y_train = np.concatenate([np.zeros(len(a_tr)), np.ones(len(b_tr))]).astype(np.int64)
    y_eval = np.concatenate([np.zeros(len(a_ev)), np.ones(len(b_ev))]).astype(np.int64)
    res, _ = _probe_pair(feats_train, y_train, feats_eval, y_eval, layers, max_iter, probe_c)
    return res


def _fmt(value):
    return "  nan " if value != value else f"{value:.4f}"


def print_table(results, layers, title):
    print(f"\n==== {title} ====")
    header = "layer | " + " | ".join(f"{kind:>6}" for kind in MODULES)
    print(header)
    print("-" * len(header))
    for layer_idx in layers:
        mark = "*" if layer_idx in HIGHLIGHT_LAYERS else " "
        row = f"{mark}{layer_idx:>4} | " + " | ".join(
            f"{_fmt(results[kind][layer_idx]):>6}" for kind in MODULES
        )
        print(row)


def _module_layer_best(results, layers, kind, maximize=True):
    vals = [
        (li, results[kind][li])
        for li in layers
        if results[kind][li] == results[kind][li]
    ]
    if not vals:
        return None, float("nan")
    return (max if maximize else min)(vals, key=lambda t: t[1])


def _mean_at(results, kind, picks):
    vals = [
        results[kind][li]
        for li in picks
        if li in results[kind] and results[kind][li] == results[kind][li]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(mode_label, id_res, dom_droid, dom_aha, layers, grounding=None):
    picks = [li for li in HIGHLIGHT_LAYERS if li in layers]
    final = layers[-1]

    print(f"\n==== {mode_label} summary ====")
    best_kind, best_layer, best = None, None, -1.0
    for kind in MODULES:
        li, val = _module_layer_best(id_res, layers, kind, maximize=True)
        if val == val and val > best:
            best_kind, best_layer, best = kind, li, val
    print(
        f"[ID failure | CALVIN] best: {best_kind} @ layer {best_layer} = {_fmt(best)}"
    )
    print(
        f"  intermediate Act@{{6,12,18}} mean = {_fmt(_mean_at(id_res, 'Act', picks))}, "
        f"final RC2 = {_fmt(id_res['RC2'][final])}"
    )
    act_int = _mean_at(id_res, "Act", picks)
    rc2_fin = id_res["RC2"][final]
    act_beats_final = (
        act_int == act_int and rc2_fin == rc2_fin and act_int > rc2_fin
    )
    print(
        f"  intermediate Act vs final RC2: "
        f"{'Act wins' if act_beats_final else 'final RC2 wins or tie'}"
    )

    if grounding is not None:
        g_act = _mean_at(grounding, "Act", picks)
        g_rc2 = grounding["RC2"][final]
        print(
            f"[Grounding sensitivity | CALVIN test] mean |Δp| when task swapped: "
            f"Act@{{6,12,18}} = {_fmt(g_act)}, final RC2 = {_fmt(g_rc2)} "
            f"(higher = more task-sensitive readout)"
        )

    summary = {
        "id_failure_best": {"module": best_kind, "layer": best_layer, "auroc": best},
        "id_failure_act_intermediate_mean": act_int,
        "id_failure_rc2_final": rc2_fin,
        "act_intermediate_beats_final_rc2": bool(act_beats_final),
        "domain_calvin_vs_droid": {
            "act_intermediate_mean": _mean_at(dom_droid, "Act", picks),
            "rc2_final": dom_droid["RC2"][final],
        },
        "domain_calvin_vs_aha": {
            "act_intermediate_mean": _mean_at(dom_aha, "Act", picks),
            "rc2_final": dom_aha["RC2"][final],
        },
    }
    if grounding is not None:
        summary["grounding_sensitivity"] = {
            "act_intermediate_mean": g_act,
            "rc2_final": g_rc2,
        }
    return summary


def _results_to_json(results, layers):
    return {kind: {str(li): results[kind][li] for li in layers} for kind in MODULES}


def _run_one_mode(mode, args, extractor, layers, calvin_train, calvin_test, droid_raw, aha_raw):
    projector = None
    if mode == "text_guided":
        projector = FixedTextGuidedProjector(
            extractor.module_dims,
            extractor.query_dim,
            args.key_dim,
            args.seed,
        )
        print(f"Text-guided pooling: mode={args.text_pool_mode}, key_dim={args.key_dim}")

    print(f"Extracting CALVIN train ({mode})...")
    if mode == "text_guided":
        feats_train, y_train, _ = extract_split(
            extractor, calvin_train, "calvin-train", mode, projector, args.text_pool_mode
        )
    else:
        feats_train, y_train, _ = extract_split(
            extractor, calvin_train, "calvin-train", mode
        )

    print(f"Extracting CALVIN test ({mode})...")
    if mode == "text_guided":
        feats_id, y_id, task_list = extract_split(
            extractor,
            calvin_test,
            "calvin-test",
            mode,
            projector,
            args.text_pool_mode,
        )
    else:
        feats_id, y_id, task_list = extract_split(
            extractor, calvin_test, "calvin-test", mode
        )

    print(f"Extracting DROID ({mode})...")
    if mode == "text_guided":
        feats_droid, _, _ = extract_split(
            extractor, droid_raw, "droid", mode, projector, args.text_pool_mode
        )
    else:
        feats_droid, _, _ = extract_split(extractor, droid_raw, "droid", mode)

    print(f"Extracting AHA ({mode})...")
    if mode == "text_guided":
        feats_aha, _, _ = extract_split(
            extractor, aha_raw, "aha", mode, projector, args.text_pool_mode
        )
    else:
        feats_aha, _, _ = extract_split(extractor, aha_raw, "aha", mode)

    print("Fitting CALVIN ID failure probes...")
    id_res, models = _probe_pair(
        feats_train, y_train, feats_id, y_id, layers, args.max_iter, args.probe_C
    )
    print("Fitting domain-invariance probes...")
    dom_droid = probe_domain(
        feats_train, feats_droid, layers, args.max_iter, args.probe_C, args.seed
    )
    dom_aha = probe_domain(
        feats_train, feats_aha, layers, args.max_iter, args.probe_C, args.seed
    )

    del feats_train, feats_id, feats_droid, feats_aha
    gc.collect()

    grounding = None
    if mode == "text_guided":
        print("Measuring grounding sensitivity (task query swap on CALVIN test)...")
        grounding = probe_grounding_sensitivity(
            calvin_test,
            task_list,
            models,
            layers,
            projector,
            args.text_pool_mode,
            extractor,
            args.seed,
        )
        print_table(
            grounding, layers, "GROUNDING sensitivity |Δp| (task swap, higher=more sensitive)"
        )

    tag = "vision-only" if mode == "vision_only" else "text-guided"
    print_table(id_res, layers, f"CALVIN ID FAILURE AUROC ({tag})")
    print_table(dom_droid, layers, f"DOMAIN CALVIN vs DROID ({tag}, lower=invariant)")
    print_table(dom_aha, layers, f"DOMAIN CALVIN vs AHA ({tag}, lower=invariant)")
    summary = summarize(
        mode_label=tag,
        id_res=id_res,
        dom_droid=dom_droid,
        dom_aha=dom_aha,
        layers=layers,
        grounding=grounding,
    )

    return {
        "id_failure_auroc": _results_to_json(id_res, layers),
        "domain_calvin_vs_droid_auroc": _results_to_json(dom_droid, layers),
        "domain_calvin_vs_aha_auroc": _results_to_json(dom_aha, layers),
        "grounding_sensitivity": (
            _results_to_json(grounding, layers) if grounding is not None else None
        ),
        "summary": summary,
    }


def main():
    args = build_parser().parse_args()
    os.makedirs(args.result_folder, exist_ok=True)
    num_entry = _parse_num_entry(args.num_entry)

    modes = []
    if args.probe_mode in ("vision_only", "both"):
        modes.append("vision_only")
    if args.probe_mode in ("text_guided", "both"):
        modes.append("text_guided")

    print("Loading splits...")
    calvin_train = load_data(
        dataset_name="calvin",
        style=args.style,
        pov=args.pov,
        split="train",
        num_entry=num_entry,
        seed=args.seed,
    )
    calvin_test = load_data(
        dataset_name="calvin",
        style=args.style,
        pov=args.pov,
        split="test",
        num_entry="full",
        seed=args.seed,
    )
    droid_raw = load_data(
        dataset_name="droid",
        style=args.style,
        pov=args.pov,
        split="test",
        num_entry="full",
        seed=args.seed,
    )
    aha_raw = load_data(
        dataset_name="aha",
        style=args.style,
        pov=args.pov,
        split="test",
        num_entry="full",
        seed=args.seed,
    )

    print("Building feature extractor (vision forward only; queries from embed table)...")
    extractor = VisionFeatureExtractor(
        model_id=args.vlm_model_id,
        revision=args.revision,
        device=args.device,
        max_pixels=args.max_pixels,
        layers=args.layers,
    )
    layers = extractor.layers
    print(
        f"Probing {len(layers)} layers x {len(MODULES)} modules "
        f"(Act dim={extractor.module_dims['Act']}, "
        f"LN2/RC2 dim={extractor.module_dims['LN2']})."
    )

    out_path = os.path.join(args.result_folder, "probe_results.json")
    all_results = {}
    if args.merge_existing_results and os.path.exists(out_path):
        with open(out_path) as handle:
            existing = json.load(handle)
        all_results = existing.get("results", {})
        print(f"Merging into existing results at {out_path} (modes: {list(all_results)})")

    for mode in modes:
        print(f"\n{'=' * 60}\nRunning probe mode: {mode}\n{'=' * 60}")
        all_results[mode] = _run_one_mode(
            mode, args, extractor, layers, calvin_train, calvin_test, droid_raw, aha_raw
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extractor.close()

    out = {
        "config": vars(args),
        "layers": layers,
        "modules": list(MODULES),
        "results": all_results,
    }
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nSaved probe results to {out_path}")


if __name__ == "__main__":
    main()
