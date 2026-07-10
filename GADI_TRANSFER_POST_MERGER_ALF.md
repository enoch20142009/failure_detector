# GADI transfer record — post-merger ALF fixes (Jul 2026)

Apply these changes on the GADI codespace before re-running job 31
(`31_train_eval_post_merger_alf.sh`). All paths are relative to repo root
`I-FailSense-main/`.

---

## Summary

Three related improvements for `--use_post_merger_alf`:

1. **Merger-input parity** — intermediate ViT depths use the same
   `layernorm_post` preprocessing as the native base path (no-op on OV2-8B where
   `use_head=false`, but correct for other checkpoints).
2. **ALF depth entropy regularization** — reuse existing `--layer_balance_coef`
   to penalize collapsed ALF depth attention (`last_alpha`), matching NGF.
3. **Smoke script** — cheap local sanity check before full PBS jobs.

---

## 1. `src/model_ov2_routed_matca.py`

### 1a. `PostMergerCrossLayerFusion` — add entropy aux loss

**`__init__`**: add `layer_balance_coef=0.0` parameter and store it; add
`self.last_aux_loss = None`.

**`forward`**: after `attn = softmax(...)`:

```python
mean_alpha = attn.mean(dim=0)          # [L], keep grad
self.last_alpha = mean_alpha.detach()
if self.layer_balance_coef > 0:
    ma = mean_alpha.clamp(min=1e-9)
    self.last_aux_loss = self.layer_balance_coef * (ma * ma.log()).sum()
else:
    self.last_aux_loss = None
```

(Same formula as NGF `NestedGuidedFusion`.)

### 1b. Wire aux loss into training

**`_PostMergerALFMerger.forward`**: after `fusion(...)` call:

```python
parent._aux_loss = fusion.last_aux_loss
```

The existing train loop already does `loss = loss + model._aux_loss` when set.

### 1c. Pass coef when building fusion module

In `OV2RoutedMaTCA.__init__`, where `PostMergerCrossLayerFusion` is created:

```python
self.post_merger_fusion = PostMergerCrossLayerFusion(
    feature_dim=self.feature_dim,
    query_dim=self.feature_dim,
    num_layers=len(self.vision_layer_indices),
    router_dim=alf_router_dim,
    layer_balance_coef=self.layer_balance_coef,
)
```

### 1d. Merger-input parity helper (from prior session)

Add method on `OV2RoutedMaTCA`:

```python
def _prepare_merger_vision_tokens(self, tokens, dtype=None):
    if dtype is not None:
        tokens = tokens.to(dtype)
    layernorm_post = getattr(self.vision_model, "layernorm_post", None)
    if layernorm_post is not None:
        tokens = layernorm_post(tokens)
    return tokens
```

In `_PostMergerALFMerger.forward`, intermediate merger inputs use:

```python
parent._prepare_merger_vision_tokens(hidden_states[idx], dtype=x.dtype)
```

instead of raw `hidden_states[idx].to(x.dtype)`.

### 1e. Epoch logging — collapse guard

In `train_model` end-of-epoch block for `post_merger_fusion`, after printing
`ALF depth attn`, add:

```python
if max(alpha_weights) > 0.9:
    print("  [collapse-guard] ALF max(depth attn) > 0.9 — consider raising --layer_balance_coef")
```

---

## 2. `src/finetune_FS_ov2_routed.py`

Update `--layer_balance_coef` help text to mention post-merger ALF (no logic
change — flag already passed to `OV2RoutedMaTCA`).

---

## 3. Training / PBS scripts

Add to post-merger ALF train invocations (default `0.01`, same as NGF smoke):

```bash
--layer_balance_coef "${LAYER_BALANCE_COEF:-0.01}" \
```

**Files updated locally:**

| File | Purpose |
|------|---------|
| `gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh` | Full GADI train+eval |
| `scripts/a6000_smoke_post_merger_alf.sh` | Local 500-sample smoke |
| `scripts/a6000_overnight_post_merger_alf.sh` | Local overnight train |

Override at submit time, e.g.:

```bash
LAYER_BALANCE_COEF=0.02 qsub gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh
```

---

## 4. No changes required

- `src/evaluate_FS_ov2_routed.py` — already saves/loads `layer_balance_coef` in
  config; aux loss is train-only.
- `POST_MERGER_ALF.md` — optional doc refresh only.

---

## 5. Recommended GADI command (job 31)

After syncing code:

```bash
# optional: override depth entropy strength
export LAYER_BALANCE_COEF=0.01
export VISION_LAYERS="6 12 18"
qsub gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh
```

Watch training log for:

```
ALF depth attn (layers [6, 12, 18]) = [...]
aux=...   # should be non-zero when layer_balance_coef > 0
```

Healthy depth spread: no single layer > 0.9 (collapse-guard warning if it is).

---

## 6. Local smoke (before full job)

```bash
source env_a6000.sh
bash scripts/a6000_smoke_post_merger_alf.sh
# log: smoke_post_merger_alf.log
# results: results_ov2_smoke_post_merger_alf/
```

Defaults: 500 CALVIN samples, 2 epochs, `layer_balance_coef=0.01`.

---

## 7. Git-style diff checklist

```
M  src/model_ov2_routed_matca.py
M  src/finetune_FS_ov2_routed.py
M  gadi_scripts/ov2_routing/31_train_eval_post_merger_alf.sh
A  scripts/a6000_smoke_post_merger_alf.sh
A  scripts/a6000_smoke_m0.sh
A  scripts/a6000_train_post_merger_alf_v2.sh
A  scripts/a6000_chain_step2_eval_step3.sh
A  scripts/a6000_sequence_post_merger_alf.sh
M  scripts/a6000_overnight_post_merger_alf.sh
A  GADI_TRANSFER_POST_MERGER_ALF.md
```

Prior session (if not yet on GADI): JSON fix in `src/evaluate_FS_ov2_routed.py`
for `_to_json_serializable()` — needed for eval after train.

---

## 8. Stage-1 higher LR (`--stage1_lr`) — step 3 fallback

If full run still shows `beta`/`gamma` ~0 and val_acc ~0.5, use separate LRs:

```bash
--lr 1e-4 --stage1_lr 3e-4 --layer_balance_coef 0.001
```

**Code:** `OV2RoutedMaTCA.optimizer_param_groups()` splits Stage-1 modules
(`post_merger_fusion`, `post_merger_adapter`, router, NGF, etc.) from MaTCA head.

**GADI / local v2 script:** `scripts/a6000_train_post_merger_alf_v2.sh`

---

## 9. Experiment sequence (aloha-server)

| Step | Script | What |
|------|--------|------|
| 1 | `scripts/a6000_smoke_m0.sh` | M0 baseline smoke (500 samples) |
| 2 | `scripts/a6000_overnight_post_merger_alf.sh` | Full train + `layer_balance_coef=0.01` |
| 2b | `scripts/a6000_chain_step2_eval_step3.sh` | Auto eval + conditional step 3 |
| 3 | `scripts/a6000_train_post_merger_alf_v2.sh` | `stage1_lr=3e-4`, `layer_balance_coef=0.001` |

Or run all: `nohup bash scripts/a6000_sequence_post_merger_alf.sh >> sequence.log 2>&1 &`
