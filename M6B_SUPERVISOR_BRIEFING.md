# M6b Champion — Supervisor Meeting Brief

**Method ID:** M6b (Contrastive Outer-Only NGF + Base Residual, COD-α + η)  
**Checkpoint:** `results_ov2_calvin_m6_cod_alpha_residual`  
**Full doc:** [`M6B_ARCHITECTURE.md`](M6B_ARCHITECTURE.md)

---

## One-liner

Hook 4 ViT depths, pick depth weights with **contrastive task−fail routing**, fuse into `F`, then **blend into the native merger field** via `V_out = V_base + η(F − V_base)` before frozen merger → frozen LLM → MaTCA head.

---

## Results (CALVIN train → eval)

| Method | CALVIN AUROC | **DROID AUROC** | Key idea |
|--------|-------------:|----------------:|----------|
| **M6b (champion)** | **0.976** | **0.831** | Multilayer contrastive + η-blend anchor |
| M0 flat router | 0.979 | 0.823 | Single-layer contrastive @ V_base |
| M6a replace | 0.974 | 0.819 | Same as M6b but `F` replaces V_base |
| Category agg residual | 0.973 | 0.811 | 6×4 category router (newer, not champion) |

Primary metric: **CALVIN → DROID AUROC** (cross-domain transfer).

Eval JSON:
- `eval_results/ov2_calvin_m6_cod_alpha_residual_indomain/results.json`
- `eval_results/ov2_calvin_m6_cod_alpha_residual_to_droid/results.json`

---

## End-to-end pipeline

**Legend**

| Tag | Meaning |
|-----|---------|
| `[OV2]` | Baseline frozen OneVision-2 path (pretrained, unchanged weights) |
| `[M6b+]` | Our trainable insertion (Stage-1 routing / blend / adapter) |
| `[OV2∥M6b+]` | Baseline module kept frozen, plus a parallel trainable branch |
| `[FailSense]` | Task-specific trainable head (not part of stock OV2 chat inference) |

Baseline OV2 alone would be: **Image → ViT → V_base → Merger → LLM**.  
M6b **intercepts** before the merger, blends multilayer `F` into `V_base`, then re-enters the baseline merger+LLM path.

```text
Image + task text
    │
    ▼
[OV2] ViT encoder (24 blocks, frozen)
    │ hook hidden_states at layers 6, 12, 18, 24
    │                              └── V_base (layer 24) is native OV2 merger input
    │
    ▼
[M6b+] Contrastive depth router  →  α over 4 depths
    │
    ▼
[M6b+] Fuse: F = Σ_l α_l · V_l        (1024-d patch tokens)
    │
    ▼
[M6b+] η-blend: V_out = V_base + η(F − V_base)   ← champion change (η init 0 → starts as OV2)
    │
    ▼
[OV2∥M6b+] Merger_frozen(V_out) + γ·Adapter(V_out)   1024 → 4096 LLM tokens
    │         └── OV2 patch merger (frozen)  +  parallel adapter (trainable, γ init 0)
    │
    ▼
[OV2] LLaVA-OneVision-2 8B LLM (frozen)
    │
    ▼
[FailSense] MaTCA head @ LM layers 19, 28, 36  →  fail/success logit
```

**Dimensions:**
- Pre-merger vision tokens: **1024-d** (`vision_dim`)
- Post-merger LLM tokens: **4096-d** (`feature_dim`)
- Text queries `t_task`, `t_fail`: **4096-d** (frozen LM embeddings, mean-pooled)

```1519:1520:src/model_ov2_routed_matca.py
        self.vision_dim = vision_config.hidden_size          # 1024
        self.feature_dim = text_config.hidden_size           # 4096 (LM hidden)
```

---

## Math (4 stages)

### Stage 0 — Inputs

- `V_l ∈ R^{B×N×1024}` for `l ∈ {6, 12, 18, base}` — hooked ViT patch fields
- `V_base` = layer 24 = native merger input
- `t_task`, `t_fail ∈ R^{B×4096}` — frozen text embeddings (task vs failure template)

### Stage 1 — Contrastive depth weights (outer loop only)

Pool each depth over patches, score with task vs fail, softmax:

\[
\mu_l = \frac{1}{N}\sum_n V_{l,n}
\]

\[
K_l = W_K \mu_l, \quad s_{\text{task},l} = \frac{\langle W_{Qt} t_{\text{task}}, K_l \rangle}{\sqrt{256}}, \quad s_{\text{fail},l} = \frac{\langle W_{Qf} t_{\text{fail}}, K_l \rangle}{\sqrt{256}}
\]

\[
\alpha_l = \text{softmax}_l\big(s_{\text{task},l} - s_{\text{fail},l}\big)
\]

**Grounding rule:** text only produces `α_l`; the fused field is purely visual.

```373:397:src/model_ov2_routed_matca.py
class ContrastiveDepthRouter(nn.Module):
    """Depth-level task vs failure contrastive weights (Option A / M6).
    ...
    """
    ...
    def forward(self, stacked, t_task, t_fail):
        mu = stacked.mean(dim=2)                              # [B, L, Dv]
        keys = self.w_k(mu)                                   # [B, L, Dr]
        ...
        logits = s_task - s_fail
        return torch.softmax(logits, dim=-1)
```

M6b sets **inner loop off** → `h_l = V_l` (no per-patch gating):

```690:691:src/model_ov2_routed_matca.py
            else:
                h_l = v_l
```

### Stage 2 — Depth fusion

Same `α_l` for every patch `n`:

\[
F_n = \sum_{l} \alpha_l \cdot V_{l,n}
\]

```708:716:src/model_ov2_routed_matca.py
            if self.layer_weight_mode == "contrastive":
                alpha = self.layer_router(stacked, t_task, t_fail)
            ...
            fused = torch.sum(stacked * alpha[:, :, None, None], dim=1)  # [B, N, Dv]
```

### Stage 3 — M6b base-anchored blend (the champion change)

\[
V_{\text{out}} = V_{\text{base}} + \eta \cdot (F - V_{\text{base}}) = (1-\eta)\,V_{\text{base}} + \eta\,F
\]

- `η` init **0** → starts as identity (`V_out = V_base`)
- `η → 1` → equivalent to M6a full replace

```648:649:src/model_ov2_routed_matca.py
        if not replace_base:
            self.eta = nn.Parameter(torch.zeros(1))
```

```726:728:src/model_ov2_routed_matca.py
        if self.replace_base:
            return fused
        return v_base + self.eta * (fused - v_base)
```

Enabled via `--nested_residual` → `nested_replace_base=False`:

```140:141:src/finetune_FS_ov2_routed.py
    parser.add_argument("--nested_residual", action="store_true",
                        help="Blend fused field as V_base + eta*(F - V_base), eta init 0, "
```

```294:294:src/finetune_FS_ov2_routed.py
        nested_replace_base=not args.nested_residual,
```

### Stage 4 — Merger + parallel adapter

\[
H = \text{Merger}_{\text{frozen}}(V_{\text{out}}) + \gamma \cdot \text{Adapter}(V_{\text{out}})
\]

- Merger: **1024 → 4096** (frozen)
- Adapter: **1024 → 64 → 4096** (trainable, `γ` init 0)

```1064:1082:src/model_ov2_routed_matca.py
class MergerAdapter(nn.Module):
    """Low-rank parallel path on the OV2 patch merger (native weights stay frozen).
        h = Merger_frozen(x) + γ · Adapter(x),   γ init = 0
    """
    ...
    def forward(self, x):
        merged = self.ln_q(x32).reshape(-1, self.hidden_size)
        return self.up(F.gelu(self.down(merged))).to(out_dtype)
```

Wired in merger wrapper:

```1115:1120:src/model_ov2_routed_matca.py
    def forward(self, x, patch_positions=None):
        parent = self._parent_ref[0]
        x = parent.apply_upstream(x)
        h_base = self.merger(x, patch_positions=patch_positions)
        h_adapt = self.adapter(x)
        return h_base + self.adapter.gamma * h_adapt.to(h_base.dtype)
```

Upstream fusion runs inside `apply_upstream` before merger:

```1909:1951:src/model_ov2_routed_matca.py
        if self.use_nested_guided_fusion:
            ...
                layer_tokens = [
                    hidden_states[idx].float() for idx in self.vision_layer_indices
                ]
            ...
            x32 = self.nested_fusion(
                layer_tokens, x32, self._task_query, self._fail_query
            )
            ...
            return x32.to(orig_dtype)
```

---

## What's frozen vs trainable

| Component | Tag | Status |
|-----------|-----|--------|
| OV2 ViT encoder | `[OV2]` | Frozen |
| ViT hidden_states hooks (6, 12, 18, 24) | `[OV2]` read-only | Frozen features |
| `ContrastiveDepthRouter` (W_K, W_Qt, W_Qf) | `[M6b+]` | **Trainable** |
| Depth fusion + scalar `η` | `[M6b+]` | **Trainable** (η init 0) |
| Patch merger weights | `[OV2]` | Frozen |
| `MergerAdapter` + `γ` | `[M6b+]` | **Trainable** (γ init 0) |
| OV2 LLM (8B) | `[OV2]` | Frozen |
| MaTCA head (pooling + classifiers) | `[FailSense]` | **Trainable** |
| Text embeddings (t_task, t_fail) | `[OV2]` | Frozen (queries only) |

---

## Why M6b beats M6a and M0

| vs | Delta DROID AUROC | Insight |
|----|-------------------|---------|
| M6a (replace `F`) | +0.012 | Frozen merger expects `V_base`-like stats; full replace hurts transfer |
| M0 (flat single-layer router) | +0.008 | Multilayer contrastive depth routing helps, with anchor |
| Inner-on NGF | +0.028 | Per-patch inner + outer text routing is redundant / overfits CALVIN |

Observed depth weights ≈ uniform `[0.25, 0.25, 0.25, 0.25]` — router spreads across depths, no collapse.

---

## CLI flags (reproduce champion)

```text
--use_nested_guided_fusion
--ngf_no_inner                          # inner off
--ngf_layer_weight_mode contrastive     # task−fail outer
--vision_layer_indices 6 12 18
--nested_residual                       # M6b η-blend
--use_merger_adapter
--layer_balance_coef 0.01
```

Script:

```bash
NESTED_RESIDUAL=1 bash scripts/a6000_overnight_m6_cod_alpha.sh
```

---

## Paper one-liner (ready to say)

> We inject task–failure contrastive signals once at ViT depth granularity, fuse four depth streams into `F`, and blend into the native merger field via `V_base + η(F − V_base)` so the frozen connector stays anchored. On CALVIN→DROID transfer, M6b achieves DROID AUROC **0.831**, beating flat contrastive routing (0.823) and full replacement (0.819).

---

## Module map (quick code lookup)

| Component | File | Lines |
|-----------|------|-------|
| `ContrastiveDepthRouter` | `src/model_ov2_routed_matca.py` | 373–397 |
| `NestedGuidedFusion` (η blend) | same | 561–728 |
| `MergerAdapter` | same | 1064–1082 |
| `_AdaptedRoutedMerger` | same | 1106–1120 |
| `apply_upstream` (NGF hook) | same | 1909–1951 |
| Train CLI flags | `src/finetune_FS_ov2_routed.py` | 127–141, 292–294 |
| Eval results | `eval_results/ov2_calvin_m6_cod_alpha_residual_to_droid/results.json` | auroc=0.831 |

---

## ASCII data-flow (single patch index n)

```text
         t_task, t_fail  [OV2 frozen LM embeds, used as queries only]
              │
              ▼
    ┌─────────────────────────┐
    │ ContrastiveDepthRouter  │──► α = [α_6, α_12, α_18, α_base]     [M6b+]
    └─────────────────────────┘
              │
   V_6[n] V_12[n] V_18[n] V_base[n]   ← V_* from [OV2] ViT hidden_states hook
              │
              ▼
         F[n] = Σ_l α_l · V_l[n]                                    [M6b+]
              │
              ▼
    V_out[n] = V_base[n] + η · (F[n] − V_base[n])   ◄── M6b only   [M6b+]
              │                                        (η=0 → V_out = V_base, same as OV2)
              ▼
    Merger(V_out)  +  γ·Adapter(V_out)  →  LLM  →  MaTCA
       [OV2]              [M6b+]           [OV2]    [FailSense]
```

**Baseline OV2 shortcut (no M6b):** `V_base[n] → Merger → LLM` (skip router, fusion, η-blend, adapter).
```

---

## Related method IDs

| ID | Short name |
|----|------------|
| M0 | Flat `DualQueryRouter` @ `V_base` + adapter |
| M6a | COD-α, replace `V_base` with `F` |
| **M6b** | **COD-α + η residual blend (champion)** |
| E | Inner patch ×4 + task-only outer |
