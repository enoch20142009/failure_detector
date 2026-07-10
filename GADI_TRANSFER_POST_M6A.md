# GADI transfer — post-M6a code + jobs (Jul 2026)

Apply on GADI at `/scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main`
after pulling branch **`trans_1Jul2026`**.

Companion: [`GADI_TRANSFER_POST_MERGER_ALF.md`](GADI_TRANSFER_POST_MERGER_ALF.md) (job 31 ALF fixes, already merged here).

---

## 1. Pull on GADI

```bash
cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main
git fetch origin
git checkout trans_1Jul2026
git pull origin trans_1Jul2026
chmod +x gadi_scripts/ov2_routing/{31..36}_*.sh
```

No new pip deps beyond existing `.venv-ov2` + `pytorch/2.12.0`.

---

## 2. Source code changes (since M6a)

| File | What changed |
|------|----------------|
| `src/model_ov2_routed_matca.py` | `ContrastiveDepthRouter`; `CategoryContrastiveAggregator` (6×4, concat modes); NGF `contrastive` outer; `--ngf_full_connector` also for category-agg |
| `src/finetune_FS_ov2_routed.py` | `--ngf_layer_weight_mode contrastive`; `--use_category_aggregator` + concat modes; `--stage1_lr`; category / full-connector flags |
| `src/evaluate_FS_ov2_routed.py` | category-agg config load; `category_alpha`, `category_concat_mode`, `category_agg_gamma` in results |
| `gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh` | `--layer_balance_coef` passthrough |

### New modules (high level)

- **M6 contrastive outer** — `ngf_layer_weight_mode=contrastive` → `ContrastiveDepthRouter` on depth summaries
- **M6b residual** — `--nested_residual` → `V_out = V_base + η(F − V_base)` (job 32)
- **Category aggregator** — 6 categories × 4 ViT layers; contrastive α over categories; concat modes:
  - `residual_last` — `V_base + γ·Adapter([V_base;F])` (best local cat-agg)
  - `igva_penultimate` — `Adapter([F;F_pen])`
  - `igva_base` — `Adapter([F;V_base])`
- **Trainable merger** — `--ngf_full_connector` (no `--use_merger_adapter`); warm-started merger clone

---

## 3. PBS jobs (post-M6a)

| Job | Script | Method |
|-----|--------|--------|
| **31** | `31_train_eval_post_merger_alf.sh` | Post-merger ALF |
| **32** | `32_train_eval_m6_cod_alpha.sh` | M6a replace (default) |
| **32b** | `NESTED_RESIDUAL=1 qsub …/32_…` | **M6b champion** (η residual) |
| **33** | `33_train_eval_m6_task_only_outer.sh` | M6 task-only outer ablation |
| **34** | `34_train_eval_m6_inner_on.sh` | M6 inner-on ablation |
| **35** | `35_train_eval_m6_cod_alpha_every3.sh` | M6 every-3 depths |
| **36** | `36_train_eval_category_agg.sh` | Category agg (see env below) |

### Recommended GADI queue order

```bash
# Champion replication (if not already on GADI)
NESTED_RESIDUAL=1 qsub gadi_scripts/ov2_routing/32_train_eval_m6_cod_alpha.sh

# Category agg variants (job 36 + env)
qsub gadi_scripts/ov2_routing/36_train_eval_category_agg.sh
CATEGORY_CONCAT_MODE=igva_base qsub gadi_scripts/ov2_routing/36_train_eval_category_agg.sh
CATEGORY_CONCAT_MODE=residual_last FULL_CONNECTOR=1 qsub gadi_scripts/ov2_routing/36_train_eval_category_agg.sh
```

### Job 36 env vars

| Var | Default | Values |
|-----|---------|--------|
| `CATEGORY_CONCAT_MODE` | `residual_last` | `residual_last`, `igva_penultimate`, `igva_base` |
| `FULL_CONNECTOR` | `0` | `1` = trainable merger, no MergerAdapter |

---

## 4. Local A6000 results (reference, Jul 2026)

Primary metric: **CALVIN → DROID AUROC**.

| Method | CALVIN AUROC | DROID AUROC | Notes |
|--------|-------------:|------------:|-------|
| **M6b** | 0.976 | **0.831** | Champion |
| M0 flat router | 0.979 | 0.823 | |
| M6a replace | 0.974 | 0.819 | |
| Cat-agg residual + adapter | 0.973 | 0.811 | job 36 default |
| Cat-agg igva_pen | 0.712 | 0.493 | job 36 `igva_penultimate` |
| Cat-agg igva_base | TBD | TBD | training |
| Cat-agg full connector | TBD | TBD | training |

Docs: `M6B_ARCHITECTURE.md`, `M6B_SUPERVISOR_BRIEFING.md`, `OUTER_CONTRASTIVE_DEPTH_OPTION_A.md`.

---

## 5. Result paths (match local)

| Run | `results_*` | `eval_results/*_to_droid` |
|-----|-------------|---------------------------|
| M6b | `results_ov2_calvin_m6_cod_alpha_residual` | `ov2_calvin_m6_cod_alpha_residual_to_droid` |
| M6a | `results_ov2_calvin_m6_cod_alpha_replace` | `…_replace_to_droid` |
| Cat-agg residual | `results_ov2_calvin_category_agg` | `ov2_calvin_category_agg_to_droid` |
| Cat-agg igva | `results_ov2_calvin_category_agg_igva` | `…_igva_to_droid` |
| Cat-agg igva_base | `results_ov2_calvin_category_agg_igva_base` | `…_igva_base_to_droid` |
| Cat-agg full conn | `results_ov2_calvin_category_agg_full_connector` | `…_full_connector_to_droid` |

---

## 6. Quick smoke before full PBS

```bash
source .venv-ov2/bin/activate
python -m py_compile src/model_ov2_routed_matca.py src/finetune_FS_ov2_routed.py src/evaluate_FS_ov2_routed.py

python src/finetune_FS_ov2_routed.py \
  --vlm_model_id /scratch/ka69/yc0686/models/LLaVA-OneVision-2-8B-Instruct \
  --dataset_name calvin --pov 1 --num_epochs 1 --batch_size 1 \
  --target_layer_indices 19 28 36 --num_classifiers 3 \
  --use_category_aggregator --category_concat_mode residual_last \
  --use_merger_adapter --merger_adapter_rank 64 \
  --layer_balance_coef 0.01 --result_folder ./results_ov2_smoke_category_agg
```

---

## 7. Git files in this transfer

**Modified:** `src/model_ov2_routed_matca.py`, `src/finetune_FS_ov2_routed.py`, `src/evaluate_FS_ov2_routed.py`, `gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh`

**New GADI jobs:** `32` (existing), `33`, `34`, `35`, `36`

**New docs (optional on GADI):** `M6B_*.md`, `M6A_*.md`, `OUTER_CONTRASTIVE_DEPTH_OPTION_A.md`

**Local-only (not required on GADI):** `scripts/a6000_*.sh`, `env_a6000.sh`
