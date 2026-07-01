"""OneVision-2 backbone with pre-LLM task/failure visual routing + MaTCA head.

This module implements the "combined" architecture for robot success/failure
detection on top of a frozen ``LLaVA-OneVision-2-8B-Instruct`` checkpoint, plus
an optional Mixture-of-Failure-Experts (MoE) routing stage.

Stages around the frozen multimodal backbone:

    Stage 1 (pre-LLM, NEW):
        - hierarchical fusion of several vision-encoder layers,
        - a dual task/failure query router,
        - a hierarchical per-depth dual-query router (``use_hier_router``), and
        - fuse-then-route (``use_fuse_then_route``): TGIF-style multi-depth feature
          fusion followed by one flat dual-query router (avoids per-depth delta collapse),
        - OPTIONALLY, a Mixture-of-Failure-Experts router (``use_moe``) that
          replaces the router's single residual transform ``phi`` with E small
          experts and a per-visual-token top-k gate. Experts specialize latently
          (no failure-mode labels needed); a relaxed load-balancing term
          (LTDR-inspired) lets rare-failure experts specialize, and an optional
          supervised-gate hook can align experts to named failure modes when
          ``failure_mode`` metadata exists.
      All Stage-1 effects are applied to the spatial visual tokens BEFORE the
      patch merger (the OneVision-2 projector), preserving token count/order.

    Stage 2 (post-LLM, the existing "MaTCA" head):
        - task-conditioned token pooling per selected LM hidden layer,
        - a learned (static or dynamic) layer fusion, and
        - per-layer + fused binary classifiers.

QMSA-inspired ablations (CVPR'26 QViC-MF):
    - ``gate_style="guiding"`` derives an additive attention-logit bias from the
      query->visual relevance (instead of multiplicative sigmoid gating), an
      ablation against the default multiplicative gates.
    - Grounding invariant ("blocking"): text (task/failure queries) may only
      gate; the values ``U = W_u(V)`` are purely visual, so text never writes
      content into the routed residual. This guards against "compression
      hallucination" (a monitor confirming whatever the prompt assumes). The
      ``--grounding_check`` / task-substitution probe measures prediction change
      under swapped query text.

The head modules (``TaskConditionedPooling``, ``LearnedLayerFusion``,
``DynamicLayerFusion``, ``MLP_BLOCK``, ...) are imported from
``model_qwen_multilayer_fusion`` so the post-LLM head is identical to the
component used by the current Qwen3-VL system.
"""

import copy
import gc
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# The OneVision-2 checkpoint registers custom processor/model code, so the
# generic Auto* classes resolve to that implementation via ``trust_remote_code``.
from transformers import AutoModelForImageTextToText, AutoProcessor

# Reuse the OneVision-2 prompt construction and label normalization so the text
# formatting matches the raw OV2 baseline exactly.
from model_ov2_baseline import (
    DEFAULT_OV2_MODEL_ID,
    DEFAULT_OV2_REVISION,
    build_messages,
    label_to_binary,
)

# Reuse the exact post-LLM head modules from the current Qwen pipeline.
from model_qwen_multilayer_fusion import (
    DynamicLayerFusion,
    HybridAttentionPooling,
    LearnedLayerFusion,
    MLP_BLOCK,
    TaskConditionedPooling,
)


# ---------------------------------------------------------------------------
# Stage 1a: hierarchical vision-layer fusion (pre-projector, identity-preserving)
# ---------------------------------------------------------------------------
class HierarchicalVisionFusion(nn.Module):
    """Fuse several vision-encoder layers into a residual update of the base layer.

    Given the configured base visual tokens ``V_base`` (the layer the merger
    normally consumes) and a list of same-shaped layer tensors ``[V_l]``, this
    module computes a fused multi-depth representation and adds it back as a
    zero-initialized residual:

        V_multi = sum_l w_l * V_l
        V_hier  = V_base + alpha * transform(V_multi),   alpha initialized to 0

    At initialization ``alpha = 0`` so ``V_hier == V_base`` and the pretrained
    vision-language interface is preserved; training only introduces hierarchical
    features when they reduce the loss.
    """

    def __init__(
        self, vision_dim, num_layers, fusion_mode="static", hidden_dim=256, query_dim=None
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.num_layers = num_layers

        # Depth-mixing weights.
        if fusion_mode == "static":
            # One global weight per layer, shared across samples and patches.
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        elif fusion_mode == "dynamic":
            # Sample-adaptive weights produced from each layer's pooled features.
            self.scorer = nn.Sequential(
                nn.LayerNorm(vision_dim),
                nn.Linear(vision_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        elif fusion_mode == "text":
            # TGIF/IGVA-style: task+failure text routes depth weights (image-agnostic α).
            if query_dim is None:
                raise ValueError("fusion_mode='text' requires query_dim")
            self.text_scorer = nn.Sequential(
                nn.LayerNorm(query_dim * 2),
                nn.Linear(query_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_layers),
            )
        elif fusion_mode == "mean":
            # No learnable weights; plain average across layers.
            pass
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # Maps the fused multi-depth features into a residual delta.
        self.transform = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, vision_dim),
        )

        # Identity-preserving residual scale (the key stability mechanism).
        self.alpha = nn.Parameter(torch.zeros(1))

    def _depth_weights(self, layers, t_task=None, t_fail=None):
        # ``layers`` is a list of [B, N, Dv] tensors, one per selected depth.
        if self.fusion_mode == "static":
            return torch.softmax(self.layer_logits, dim=0)
        if self.fusion_mode == "mean":
            return torch.full(
                (self.num_layers,),
                1.0 / self.num_layers,
                device=layers[0].device,
                dtype=layers[0].dtype,
            )
        if self.fusion_mode == "text":
            if t_task is None or t_fail is None:
                raise ValueError("text fusion requires t_task and t_fail")
            scorer_in = torch.cat([t_task, t_fail], dim=-1)
            return torch.softmax(self.text_scorer(scorer_in), dim=-1)
        # Dynamic: score each layer by its mean-pooled patch features.
        pooled = torch.stack([layer.mean(dim=1) for layer in layers], dim=1)
        scores = self.scorer(pooled).squeeze(-1)
        return torch.softmax(scores, dim=1)

    def forward(self, v_base, layers, t_task=None, t_fail=None):
        weights = self._depth_weights(layers, t_task=t_task, t_fail=t_fail)

        if weights.dim() == 1:
            # Static / mean: identical weights for every sample.
            v_multi = sum(weights[idx] * layers[idx] for idx in range(self.num_layers))
        else:
            # Dynamic: per-sample weights [B, L] broadcast over tokens/features.
            stacked = torch.stack(layers, dim=1)
            v_multi = torch.sum(stacked * weights[:, :, None, None], dim=1)

        delta = self.transform(v_multi)
        return v_base + self.alpha * delta


# ---------------------------------------------------------------------------
# Shared gating helper for the dual-query router (multiplicative or guiding)
# ---------------------------------------------------------------------------
class _PatchGate(nn.Module):
    """Compute per-patch relevance gates from a query against visual keys.

    Two styles:
      - ``multiplicative`` (default): g = sigmoid/softmax(query . K / sqrt(d)),
        used to scale the (purely visual) values.
      - ``guiding`` (QMSA-inspired): add the query->visual relevance as a bias to
        a learned, query-independent base attention logit, then softmax:
        g = softmax(base(V) + relevance). This injects the question as an
        additive attention bias rather than a multiplicative gate.
    """

    def __init__(self, vision_dim, router_dim, gate_type="sigmoid", gate_style="multiplicative"):
        super().__init__()
        self.gate_type = gate_type
        self.gate_style = gate_style
        self.scale = 1.0 / math.sqrt(router_dim)
        if gate_style == "guiding":
            # Learned query-independent base attention logit per patch.
            self.base_score = nn.Linear(vision_dim, 1)

    def forward(self, query, keys, values_v):
        # query: [B, d]; keys: [B, N, d]; values_v: [B, N, Dv] (for base score).
        relevance = torch.einsum("bd,bnd->bn", query, keys) * self.scale

        if self.gate_style == "guiding":
            base = self.base_score(values_v).squeeze(-1)
            return torch.softmax(base + relevance, dim=1)

        if self.gate_type == "sigmoid":
            return torch.sigmoid(relevance)
        if self.gate_type == "softmax":
            return torch.softmax(relevance, dim=1)
        raise ValueError(f"Unknown gate_type: {self.gate_type}")


# ---------------------------------------------------------------------------
# Stage 1b: dual task/failure query router (pre-projector, identity-preserving)
# ---------------------------------------------------------------------------
class DualQueryRouterCore(nn.Module):
    """Shared dual-query routing core: streams + phi delta, no residual scale.

    Factored out so flat ``DualQueryRouter``, hierarchical depth fusion, and MoE
    can reuse the same gated evidence and ``phi`` transform.
    """

    def __init__(
        self,
        vision_dim,
        query_dim,
        router_dim=256,
        router_mode="contrastive",
        gate_type="sigmoid",
        gate_style="multiplicative",
        share_query=False,
    ):
        super().__init__()
        self.router_mode = router_mode
        self.gate_type = gate_type
        self.gate_style = gate_style
        self.share_query = share_query

        self.w_k = nn.Linear(vision_dim, router_dim)
        self.w_u = nn.Linear(vision_dim, vision_dim)

        self.w_qt = nn.Linear(query_dim, router_dim)
        if share_query:
            self.w_qf = self.w_qt
        else:
            self.w_qf = nn.Linear(query_dim, router_dim)

        self.gate = _PatchGate(vision_dim, router_dim, gate_type, gate_style)

        self.num_streams = {"task_only": 2, "task_fail": 3, "contrastive": 4}[router_mode]
        self.phi = nn.Sequential(
            nn.LayerNorm(self.num_streams * vision_dim),
            nn.Linear(self.num_streams * vision_dim, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim),
        )

    def build_streams(self, v, t_task, t_fail):
        """Return (streams_concat, gates) without applying phi."""
        keys = self.w_k(v)
        values = self.w_u(v)

        g_t = self.gate(self.w_qt(t_task), keys, v)
        streams = [v, g_t.unsqueeze(-1) * values]

        g_f = None
        if self.router_mode in ("task_fail", "contrastive"):
            g_f = self.gate(self.w_qf(t_fail), keys, v)
            streams.append(g_f.unsqueeze(-1) * values)
        if self.router_mode == "contrastive":
            streams.append((g_t - g_f).unsqueeze(-1) * values)

        gates = {"g_task": g_t.detach(), "g_fail": None if g_f is None else g_f.detach()}
        return torch.cat(streams, dim=-1), gates

    def compute_delta(self, v, t_task, t_fail):
        """Apply phi to the gated evidence streams; no beta residual."""
        streams, gates = self.build_streams(v, t_task, t_fail)
        return self.phi(streams), gates


class DualQueryRouter(nn.Module):
    """Modulate spatial visual tokens with task and task-specific failure queries.

    Keys/values come from the visual tokens; queries come from the (frozen) LM
    embeddings of the task text and a failure template. Patch relevance gates the
    value update, which is added back as a zero-initialized residual:

        K = W_K V,            U = W_U V          (U is PURELY visual: grounding)
        q_t = W_Qt t_task,    q_f = W_Qf t_fail
        g_t, g_f = gate(q, K, V)                 (multiplicative or guiding)
        Delta = phi([V; g_t*U; g_f*U; (g_t-g_f)*U])
        V_routed = V + beta * Delta,   beta initialized to 0

    ``router_mode`` controls how much evidence is used (task_only / task_fail /
    contrastive). The text only ever forms scalar gates; it never writes content
    into ``Delta`` (the "blocking"/grounding invariant).
    """

    def __init__(
        self,
        vision_dim,
        query_dim,
        router_dim=256,
        router_mode="contrastive",
        gate_type="sigmoid",
        gate_style="multiplicative",
        share_query=False,
    ):
        super().__init__()
        self.core = DualQueryRouterCore(
            vision_dim=vision_dim,
            query_dim=query_dim,
            router_dim=router_dim,
            router_mode=router_mode,
            gate_type=gate_type,
            gate_style=gate_style,
            share_query=share_query,
        )
        self.beta = nn.Parameter(torch.zeros(1))

        # Scratch attributes set by ``forward`` / the MoE path before streaming.
        self._t_task = None
        self._t_fail = None

    @property
    def num_streams(self):
        return self.core.num_streams

    def build_streams(self, v):
        """Return (streams_concat, gates) without applying phi or the residual.

        Factored out so the MoE router can reuse the exact same gated, purely
        visual evidence streams. ``self._t_task`` / ``self._t_fail`` must be set.
        """
        return self.core.build_streams(v, self._t_task, self._t_fail)

    def forward(self, v, t_task, t_fail):
        self._t_task = t_task
        self._t_fail = t_fail
        delta, gates = self.core.compute_delta(v, t_task, t_fail)
        return v + self.beta * delta, gates


# ---------------------------------------------------------------------------
# Stage 1d: nested guided fusion (per-layer dual-query guiding + text layer router)
# ---------------------------------------------------------------------------
class TextLayerRouter(nn.Module):
    """TGIF-style outer loop: map task-text embedding to a softmax over depths.

    ``alpha = softmax(MLP(f_task)) in R^L``. Task text only (no failure query,
    no image), so the layer mixture depends on the question, not the patches.
    """

    def __init__(self, query_dim, num_layers, hidden_dim=256):
        super().__init__()
        self.num_layers = num_layers
        self.mlp = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )

    def forward(self, t_task):
        return torch.softmax(self.mlp(t_task), dim=-1)


class TokenDepthRouter(nn.Module):
    """Per-patch, per-depth depth weights ``alpha_{l,n}`` (Arch B).

    Unlike ``TextLayerRouter`` (one ``alpha_l`` per depth, shared across patches),
    this scores every ``(depth l, patch n)`` from the patch feature plus a
    task-text bias, then softmaxes across depths at each patch:

        score_{l,n} = MLP_feat(LayerNorm(h_{l,n})) + W_text(t_task)_l
        alpha_{.,n} = softmax_l(score_{.,n})              # sums to 1 over depths

    Output shape ``[B, L, N]``. This lets different patches pull from different
    depths (low-level texture vs object semantics), a more expressive fusion than
    the image-agnostic text-only depth mixture.
    """

    def __init__(self, query_dim, vision_dim, num_layers, hidden_dim=256):
        super().__init__()
        self.num_layers = num_layers
        self.feat_score = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.text_bias = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )

    def forward(self, stacked, t_task):
        # stacked: [B, L, N, Dv]; t_task: [B, query_dim]
        feat_logits = self.feat_score(stacked).squeeze(-1)  # [B, L, N]
        bias = self.text_bias(t_task)                        # [B, L]
        logits = feat_logits + bias[:, :, None]              # [B, L, N]
        return torch.softmax(logits, dim=1)                  # softmax over depths


class NestedGuidedFusion(nn.Module):
    """Nested per-(layer, patch) dual-query guided fusion of vision depths.

    Inner loop (per layer l, per patch n): a SHARED ``DualQueryRouterCore``
    produces a guided residual update applied to the raw features

        Delta_l = phi([V_l ; g_task*U ; g_fail*U ; (g_task-g_fail)*U])
        h_l     = V_l + beta_l * Delta_l                  # beta_l init 0

    where the gates ``g`` are per-patch scalars and ``U = W_u(V_l)`` is purely
    visual (blocking: text never writes content into ``h_l``). ``beta_l`` is a
    per-depth scalar initialized to 0, so at init ``h_l = V_l`` (identity-
    preserving, matching ``DualQueryRouter``'s ``V + beta*Delta``). This is the
    key fix vs the original collapse where ``phi`` fully replaced ``V_l``.

    Outer loop (per layer): depth weights ``alpha_l`` mix the SAME patch index
    across depths into one fused token field

        F_n = sum_l alpha_l * h_{l,n}

    No spatial pooling inside a layer (token count ``N`` is preserved). ``alpha``
    is task-text routed (``layer_weight_mode='text'``), static learnable
    (``'static'``), uniform (``'uniform'``), or per-patch token-level routed
    (``'token'``, Arch B: ``alpha_{l,n}`` from ``TokenDepthRouter``). When
    ``inner_guiding`` is False, ``h_l = V_l`` (raw features, TGIF-style outer
    fusion only).

    With ``replace_base=True`` the fused field replaces ``V_base`` entirely. With
    ``replace_base=False`` it is blended as ``V_base + eta*(F - V_base)`` with
    ``eta`` initialized to 0 (identity-preserving residual ablation).
    """

    def __init__(
        self,
        vision_dim,
        query_dim,
        num_layers,
        router_dim=256,
        router_mode="contrastive",
        gate_type="sigmoid",
        gate_style="guiding",
        share_query=False,
        layer_weight_mode="text",
        layer_balance_coef=0.0,
        inner_guiding=True,
        replace_base=True,
        hidden_dim=256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weight_mode = layer_weight_mode
        self.layer_balance_coef = layer_balance_coef
        self.inner_guiding = inner_guiding
        self.replace_base = replace_base

        if inner_guiding:
            self.core = DualQueryRouterCore(
                vision_dim=vision_dim,
                query_dim=query_dim,
                router_dim=router_dim,
                router_mode=router_mode,
                gate_type=gate_type,
                gate_style=gate_style,
                share_query=share_query,
            )
            # Per-depth residual scale; init 0 so h_l == V_l at start (identity).
            self.inner_betas = nn.Parameter(torch.zeros(num_layers))
        else:
            self.core = None

        if layer_weight_mode == "text":
            self.layer_router = TextLayerRouter(query_dim, num_layers, hidden_dim)
        elif layer_weight_mode == "static":
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        elif layer_weight_mode == "uniform":
            pass
        elif layer_weight_mode == "token":
            self.token_router = TokenDepthRouter(
                query_dim, vision_dim, num_layers, hidden_dim
            )
        else:
            raise ValueError(f"Unknown layer_weight_mode: {layer_weight_mode}")

        if not replace_base:
            self.eta = nn.Parameter(torch.zeros(1))

        # Diagnostics / aux populated each forward.
        self.last_aux_loss = None
        self.last_alpha = None
        self.last_gate_entropy = None

    def _alpha(self, t_task, device, dtype):
        if self.layer_weight_mode == "text":
            return self.layer_router(t_task)
        if self.layer_weight_mode == "static":
            return torch.softmax(self.layer_logits, dim=0).unsqueeze(0)
        return torch.full(
            (1, self.num_layers), 1.0 / self.num_layers, device=device, dtype=dtype
        )

    def forward(self, layer_tokens, v_base, t_task, t_fail):
        """``layer_tokens`` is ``[V_l..., V_base]`` with matching patch counts."""
        if len(layer_tokens) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} depth tensors, got {len(layer_tokens)}"
            )

        n_patches = layer_tokens[0].shape[1]
        feats = []
        ent_sum = 0.0
        ent_n = 0
        for layer_idx, v_l in enumerate(layer_tokens):
            if v_l.shape[1] != n_patches:
                raise RuntimeError(
                    "Patch count mismatch across NestedGuidedFusion layers"
                )
            if self.inner_guiding:
                delta_l, gates_l = self.core.compute_delta(v_l, t_task, t_fail)
                # Identity-preserving residual: h_l = V_l + beta_l * Delta_l.
                h_l = v_l + self.inner_betas[layer_idx] * delta_l
                g_t = gates_l.get("g_task")
                if g_t is not None:
                    p = g_t.clamp(min=1e-9)
                    ent_sum += float(-(p * p.log()).sum(dim=-1).mean().item())
                    ent_n += 1
            else:
                h_l = v_l
            feats.append(h_l)

        stacked = torch.stack(feats, dim=1)  # [B, L, N, Dv]

        if self.layer_weight_mode == "token":
            # Per-patch, per-depth weights alpha_{l,n} (Arch B).
            alpha_tok = self.token_router(stacked, t_task)  # [B, L, N]
            fused = torch.sum(stacked * alpha_tok.unsqueeze(-1), dim=1)  # [B, N, Dv]
            # Mean per-depth weight for logging / aux entropy.
            mean_alpha = alpha_tok.mean(dim=(0, 2))  # [L]
            self.last_alpha = mean_alpha.detach()
            if self.layer_balance_coef > 0:
                ma = mean_alpha.clamp(min=1e-9)
                self.last_aux_loss = self.layer_balance_coef * (ma * ma.log()).sum()
            else:
                self.last_aux_loss = None
        else:
            alpha = self._alpha(t_task, layer_tokens[0].device, layer_tokens[0].dtype)
            self.last_alpha = alpha.detach().mean(dim=0)
            fused = torch.sum(stacked * alpha[:, :, None, None], dim=1)  # [B, N, Dv]
            if self.layer_balance_coef > 0 and self.layer_weight_mode != "uniform":
                mean_alpha = alpha.mean(dim=0).clamp(min=1e-9)
                # Minimize negative entropy -> maximize depth-usage entropy.
                self.last_aux_loss = self.layer_balance_coef * (mean_alpha * mean_alpha.log()).sum()
            else:
                self.last_aux_loss = None

        self.last_gate_entropy = (ent_sum / ent_n) if ent_n else None

        if self.replace_base:
            return fused
        return v_base + self.eta * (fused - v_base)


class SequentialNestedFusion(nn.Module):
    """Depth-recurrent residual refinement of vision depths (Arch C).

    Where ``NestedGuidedFusion`` is a one-shot parallel weighted sum (order
    agnostic), this is order-aware and recurrent -- the literal "nested" reading.
    A running state ``h`` starts at ``V_base`` and is progressively refined by
    each depth, where every step's update is gated by the running state:

        h = V_base                                            # identity start
        for k in depths (shallow -> deep, base last):
            Delta_k = core.compute_delta(V_k, t_task, t_fail)  # gated evidence
            q_state = state_proj(mean_n h)                     # running summary
            q_delta = delta_proj(mean_n Delta_k)
            g_k     = sigmoid(W_g([q_state ; q_delta]))        # order-aware gate
            h       = h + beta_seq[k] * g_k * Delta_k
        F = h

    ``beta_seq`` is per-depth, initialized to 0, so at init ``F = V_base``
    (identity-preserving, same safe start as the parallel v2). The shared
    ``DualQueryRouterCore`` preserves the blocking invariant (text only gates).
    Outer alpha / token routing is not used in this mode.
    """

    def __init__(
        self,
        vision_dim,
        query_dim,
        num_layers,
        router_dim=256,
        router_mode="contrastive",
        gate_type="sigmoid",
        gate_style="guiding",
        share_query=False,
        hidden_dim=256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.core = DualQueryRouterCore(
            vision_dim=vision_dim,
            query_dim=query_dim,
            router_dim=router_dim,
            router_mode=router_mode,
            gate_type=gate_type,
            gate_style=gate_style,
            share_query=share_query,
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, router_dim),
        )
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, router_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * router_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.beta_seq = nn.Parameter(torch.zeros(num_layers))

        # Diagnostics / aux populated each forward.
        self.last_aux_loss = None
        self.last_alpha = None
        self.last_gate_entropy = None
        self.last_seq_gate = None

    def forward(self, layer_tokens, v_base, t_task, t_fail):
        """``layer_tokens`` is ``[V_l..., V_base]`` with matching patch counts."""
        if len(layer_tokens) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} depth tensors, got {len(layer_tokens)}"
            )

        n_patches = layer_tokens[0].shape[1]
        h = v_base  # running state starts at the base merger field (identity).
        ent_sum = 0.0
        ent_n = 0
        gate_means = []
        for k, v_k in enumerate(layer_tokens):
            if v_k.shape[1] != n_patches:
                raise RuntimeError(
                    "Patch count mismatch across SequentialNestedFusion layers"
                )
            delta_k, gates_k = self.core.compute_delta(v_k, t_task, t_fail)
            g_t = gates_k.get("g_task")
            if g_t is not None:
                p = g_t.clamp(min=1e-9)
                ent_sum += float(-(p * p.log()).sum(dim=-1).mean().item())
                ent_n += 1
            q_state = self.state_proj(h.mean(dim=1))        # [B, router_dim]
            q_delta = self.delta_proj(delta_k.mean(dim=1))  # [B, router_dim]
            g_k = torch.sigmoid(
                self.gate_mlp(torch.cat([q_state, q_delta], dim=-1))
            )  # [B, 1]
            h = h + self.beta_seq[k] * g_k.unsqueeze(1) * delta_k  # [B, N, Dv]
            gate_means.append(g_k.detach().mean())

        self.last_gate_entropy = (ent_sum / ent_n) if ent_n else None
        self.last_seq_gate = torch.stack(gate_means) if gate_means else None  # [L]
        # Reuse last_alpha for the existing vision-weight logging path.
        self.last_alpha = self.last_seq_gate
        self.last_aux_loss = None
        return h


# ---------------------------------------------------------------------------
# Stage 1c: hierarchical per-depth dual-query router
# ---------------------------------------------------------------------------
class HierarchicalDualQueryRouter(nn.Module):
    """Apply a shared dual-query router at each vision depth, then fuse deltas.

    For each depth l in {vision_layer_indices..., base}:

        delta_l = core.compute_delta(V_l, q_task, q_fail)
        w = depth_scorer(q_task, q_fail, layer summaries)   # query-conditioned
        Delta = sum_l w_l * delta_l
        V_out = V_base + beta * Delta,   beta init 0
    """

    def __init__(
        self,
        vision_dim,
        query_dim,
        num_depths,
        router_dim=256,
        router_mode="contrastive",
        gate_type="sigmoid",
        gate_style="multiplicative",
        share_query=False,
        depth_fusion_mode="query_cond",
        hidden_dim=256,
    ):
        super().__init__()
        self.num_depths = num_depths
        self.depth_fusion_mode = depth_fusion_mode

        self.core = DualQueryRouterCore(
            vision_dim=vision_dim,
            query_dim=query_dim,
            router_dim=router_dim,
            router_mode=router_mode,
            gate_type=gate_type,
            gate_style=gate_style,
            share_query=share_query,
        )

        if depth_fusion_mode == "query_cond":
            scorer_in = query_dim * 2 + vision_dim * num_depths
            self.depth_scorer = nn.Sequential(
                nn.LayerNorm(scorer_in),
                nn.Linear(scorer_in, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_depths),
            )
        elif depth_fusion_mode == "static":
            self.depth_logits = nn.Parameter(torch.zeros(num_depths))
        else:
            raise ValueError(f"Unknown depth_fusion_mode: {depth_fusion_mode}")

        self.beta = nn.Parameter(torch.zeros(1))

    def _depth_weights(self, t_task, t_fail, layer_tokens):
        if self.depth_fusion_mode == "static":
            return torch.softmax(self.depth_logits, dim=0)

        pooled = torch.cat([layer.mean(dim=1) for layer in layer_tokens], dim=-1)
        scorer_in = torch.cat([t_task, t_fail, pooled], dim=-1)
        return torch.softmax(self.depth_scorer(scorer_in), dim=-1)

    def forward(self, v_base, layer_tokens, t_task, t_fail):
        """``layer_tokens`` is [V_l..., V_base] with matching patch counts."""
        if len(layer_tokens) != self.num_depths:
            raise ValueError(
                f"Expected {self.num_depths} depth tensors, got {len(layer_tokens)}"
            )

        n_patches = v_base.shape[1]
        deltas = []
        gates_by_depth = {}
        for idx, v_l in enumerate(layer_tokens):
            if v_l.shape[1] != n_patches:
                raise RuntimeError(
                    f"Patch count mismatch at depth index {idx}: "
                    f"expected {n_patches}, got {v_l.shape[1]}"
                )
            delta_l, gates_l = self.core.compute_delta(v_l, t_task, t_fail)
            deltas.append(delta_l)
            gates_by_depth[idx] = gates_l

        weights = self._depth_weights(t_task, t_fail, layer_tokens)

        if weights.dim() == 1:
            delta = sum(weights[idx] * deltas[idx] for idx in range(self.num_depths))
        else:
            stacked = torch.stack(deltas, dim=1)
            delta = torch.sum(stacked * weights[:, :, None, None], dim=1)

        gates = {
            "depth_weights": weights.detach(),
            "gates_by_depth": gates_by_depth,
        }
        return v_base + self.beta * delta, gates


# ---------------------------------------------------------------------------
# Stage 1b (alt): Mixture-of-Failure-Experts router
# ---------------------------------------------------------------------------
class MoEFailureRouter(nn.Module):
    """Latent Mixture-of-Failure-Experts replacement for the router's ``phi``.

    Reuses the dual-query gated evidence streams (so the grounding invariant
    holds: text only gates, ``U`` is purely visual), but routes each visual token
    through a sparse top-k mixture of E expert MLPs instead of a single ``phi``:

        streams_n in R[num_streams*Dv]              (per patch n)
        gate_logits[n, e] = W_g([V_n ; q_t ; q_f])  (per-token routing)
        top-k softmax over experts -> g_{n,e}
        Delta_n = sum_{e in topk} g_{n,e} * phi_e(streams_n)
        V_routed = V + beta * Delta,   beta init 0

    Specialization is latent (no labels needed). A relaxed Switch-style load
    balancing term (LTDR-inspired) is exposed via ``load_balance_coef`` (small or
    zero so rare-failure experts can specialize). An optional supervised gate CE
    (off by default) can align experts to named failure modes when available.
    """

    def __init__(
        self,
        vision_dim,
        query_dim,
        num_streams,
        num_experts=4,
        top_k=2,
        router_dim=256,
        load_balance_coef=0.01,
        expert_hidden=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_coef = load_balance_coef
        expert_hidden = expert_hidden or vision_dim

        in_dim = num_streams * vision_dim
        # Per-token expert gate, conditioned on the patch feature + both queries.
        self.q_proj = nn.Linear(query_dim, router_dim)
        self.qf_proj = nn.Linear(query_dim, router_dim)
        self.v_proj = nn.Linear(vision_dim, router_dim)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, num_experts),
        )

        # E small expert MLPs, each mapping the streams to a vision-dim delta.
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, expert_hidden),
                nn.GELU(),
                nn.Linear(expert_hidden, vision_dim),
            )
            for _ in range(num_experts)
        )

        # Identity-preserving residual scale.
        self.beta = nn.Parameter(torch.zeros(1))

        # Diagnostics / aux state (populated each forward).
        self.last_aux_loss = None
        self.last_expert_usage = None
        self.last_gate_logits = None

    def _gate_logits(self, v, t_task, t_fail):
        # v: [B, N, Dv]; queries: [B, Dq] broadcast over patches.
        ctx = self.v_proj(v)
        ctx = ctx + self.q_proj(t_task).unsqueeze(1) + self.qf_proj(t_fail).unsqueeze(1)
        return self.gate_head(ctx)  # [B, N, E]

    def forward(self, v, streams, t_task, t_fail):
        logits = self._gate_logits(v, t_task, t_fail)  # [B, N, E]
        self.last_gate_logits = logits.detach()

        # Top-k sparse routing per token.
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_soft = torch.softmax(topk_vals, dim=-1)  # [B, N, k]

        # Scatter the top-k weights back to a dense [B, N, E] weight tensor.
        gate_w = torch.zeros_like(logits)
        gate_w.scatter_(-1, topk_idx, topk_soft)

        # Combine expert outputs weighted by the (sparse) gate.
        delta = torch.zeros_like(v)
        for e, expert in enumerate(self.experts):
            w_e = gate_w[..., e].unsqueeze(-1)  # [B, N, 1]
            if torch.count_nonzero(w_e) == 0:
                continue
            delta = delta + w_e * expert(streams)

        # Relaxed Switch load-balance aux: E * sum_e P_e * f_e.
        # P_e = mean soft gate prob to e over all tokens;
        # f_e = fraction of tokens that routed to e (top-k membership).
        full_soft = torch.softmax(logits, dim=-1)  # [B, N, E]
        importance = full_soft.mean(dim=(0, 1))  # [E]
        routed = (gate_w > 0).float().mean(dim=(0, 1))  # [E]
        self.last_expert_usage = routed.detach()
        aux = self.num_experts * torch.sum(importance * routed)
        self.last_aux_loss = self.load_balance_coef * aux

        v_routed = v + self.beta * delta
        gates = {"expert_usage": routed.detach(), "gate_logits": logits.detach()}
        return v_routed, gates

    def supervised_gate_loss(self, failure_mode_id):
        """Optional CE aligning the (mean-pooled) gate to a named failure mode.

        ``failure_mode_id`` is an int class id; applied only when provided (i.e.
        a failed sample with metadata). Returns a scalar loss or None.
        """
        if failure_mode_id is None or self.last_gate_logits is None:
            return None
        if failure_mode_id < 0 or failure_mode_id >= self.num_experts:
            return None
        mean_logits = self.last_gate_logits.mean(dim=1)  # [B, E]
        target = torch.full(
            (mean_logits.shape[0],), int(failure_mode_id), dtype=torch.long, device=mean_logits.device
        )
        return F.cross_entropy(mean_logits, target)


# ---------------------------------------------------------------------------
# Trainable parallel adapter on the frozen vision patch merger (Stage 1.5)
# ---------------------------------------------------------------------------
class MergerAdapter(nn.Module):
    """Low-rank parallel path on the OV2 patch merger (native weights stay frozen).

        h = Merger_frozen(x) + γ · Adapter(x),   γ init = 0
    """

    def __init__(self, context_dim, hidden_size, output_dim, rank=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.ln_q = nn.LayerNorm(context_dim)
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, output_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out_dtype = x.dtype
        x32 = x.float()
        merged = self.ln_q(x32).reshape(-1, self.hidden_size)
        return self.up(F.gelu(self.down(merged))).to(out_dtype)


# ---------------------------------------------------------------------------
# Wrapper that injects the upstream stage just before the OneVision-2 merger
# ---------------------------------------------------------------------------
class _RoutedMerger(nn.Module):
    """Wraps the original patch merger so upstream routing runs before merging.

    ``parent`` is stored inside a list so it is NOT registered as a submodule
    (otherwise the whole 8B backbone would become a child of the merger).
    """

    def __init__(self, original_merger, parent):
        super().__init__()
        self.merger = original_merger
        self._parent_ref = [parent]

    def forward(self, x, patch_positions=None):
        parent = self._parent_ref[0]
        x = parent.apply_upstream(x)
        return self.merger(x, patch_positions=patch_positions)


class _AdaptedRoutedMerger(nn.Module):
    """Frozen merger + trainable parallel adapter on the projector output."""

    def __init__(self, original_merger, parent, adapter):
        super().__init__()
        self.merger = original_merger
        self.adapter = adapter
        self._parent_ref = [parent]

    def forward(self, x, patch_positions=None):
        parent = self._parent_ref[0]
        x = parent.apply_upstream(x)
        h_base = self.merger(x, patch_positions=patch_positions)
        h_adapt = self.adapter(x)
        return h_base + self.adapter.gamma * h_adapt.to(h_base.dtype)


class _ReplacingRoutedMerger(nn.Module):
    """Fully-trainable connector that REPLACES the frozen merger for the NGF path.

    The fused field ``F`` from ``apply_upstream`` is projected to LLM tokens by a
    warm-started, fully-trainable clone of the native merger (no frozen merger,
    no parallel adapter). Computation runs in the connector's dtype (float32) and
    the output is cast back to the backbone dtype.
    """

    def __init__(self, connector, parent):
        super().__init__()
        self.connector = connector
        self._parent_ref = [parent]

    def forward(self, x, patch_positions=None):
        parent = self._parent_ref[0]
        out_dtype = x.dtype
        fused = parent.apply_upstream(x)
        cdtype = next(self.connector.parameters()).dtype
        merged = self.connector(fused.to(cdtype), patch_positions=patch_positions)
        return merged.to(out_dtype)


# ---------------------------------------------------------------------------
# Post-merger ALF fusion (fuse-after-merger, ALF-style cross-attention)
# ---------------------------------------------------------------------------
class PostMergerCrossLayerFusion(nn.Module):
    """Cross-attention over per-depth merger outputs in LLM token space (4096-d).

    Each intermediate ViT depth is passed through the frozen patch merger
    *separately* to get an LLM-native token field ``H_l``; ``V_base`` gives the
    anchor ``H_base``. For every merged token ``n`` we attend from the anchor
    query over the intermediate depths and add a residual::

        Q_n     = W_q · H_base[n]
        K_l,V_l = W_k · H_l[n], W_v · H_l[n]
        logit_l = (Q_n · K_l) / sqrt(r) + <q_task, K_l> + <q_fail, K_l>
        ctx_n   = Σ_l softmax_l(logit) · V_l
        H_out   = H_base + beta · W_o(ctx)          # beta init 0 -> identity

    Text only enters the attention *logits* (never the values), matching the
    blocking rule used by the dual-query router: the residual carries visual
    evidence, gated by the task/failure queries.
    """

    def __init__(self, feature_dim, query_dim, num_layers, router_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.scale = router_dim ** 0.5

        self.W_q = nn.Linear(feature_dim, router_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, router_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_o = nn.Linear(feature_dim, feature_dim, bias=False)
        self.q_task = nn.Linear(query_dim, router_dim, bias=False)
        self.q_fail = nn.Linear(query_dim, router_dim, bias=False)
        # beta init 0 -> exact H_base passthrough at start; the residual scale is
        # the single identity gate (W_o keeps its default init so beta receives a
        # non-zero gradient and can bootstrap the inner projections).
        self.beta = nn.Parameter(torch.zeros(1))

        self.last_alpha = None  # mean per-depth attention weight, for logging.

    def forward(self, h_base, h_layers, task_query, fail_query):
        """``h_base`` [M, D]; ``h_layers`` list of L tensors [M, D]; queries [1, Q]."""
        q = self.W_q(h_base)                                   # [M, r]
        keys = torch.stack([self.W_k(h_l) for h_l in h_layers], dim=1)   # [M, L, r]
        vals = torch.stack([self.W_v(h_l) for h_l in h_layers], dim=1)   # [M, L, D]

        content = (q.unsqueeze(1) * keys).sum(dim=-1) / self.scale        # [M, L]
        qt = self.q_task(task_query).reshape(1, -1)                       # [1, r]
        qf = self.q_fail(fail_query).reshape(1, -1)
        text_bias = (keys * qt.unsqueeze(1)).sum(dim=-1) / self.scale
        text_bias = text_bias + (keys * qf.unsqueeze(1)).sum(dim=-1) / self.scale
        logits = content + text_bias                                     # [M, L]

        attn = torch.softmax(logits, dim=-1)                            # [M, L]
        ctx = (attn.unsqueeze(-1) * vals).sum(dim=1)                    # [M, D]
        delta = self.W_o(ctx)
        self.last_alpha = attn.mean(dim=0).detach()
        return h_base + self.beta * delta


class PostMergerOutputAdapter(nn.Module):
    """Low-rank residual MLP on the fused LLM tokens (4096-d analogue of MergerAdapter).

        H_out' = H_out + gamma · up(gelu(down(LN(H_out)))),   gamma init = 0
    """

    def __init__(self, feature_dim, rank=64):
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, rank)
        self.up = nn.Linear(rank, feature_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return self.up(F.gelu(self.down(self.ln(h))))


class _PostMergerALFMerger(nn.Module):
    """Fuse-after-merger wrapper: per-depth frozen merger -> ALF cross-attention.

    ``V_base`` passes through the frozen merger *unmixed* to form the anchor
    ``H_base``; each intermediate encoder ``hidden_states[idx]`` is merged
    separately and combined with a residual cross-attention. No pre-merger
    ``apply_upstream`` blending happens on this path.
    """

    def __init__(self, original_merger, parent):
        super().__init__()
        self.merger = original_merger
        self._parent_ref = [parent]

    def forward(self, x, patch_positions=None):
        parent = self._parent_ref[0]
        out_dtype = x.dtype
        h_base = self.merger(x, patch_positions=patch_positions)
        if not parent._upstream_active:
            return h_base

        hidden_states = parent._encoder_hidden_states
        if hidden_states is None:
            return h_base

        h_layers = [
            self.merger(hidden_states[idx].to(x.dtype), patch_positions=patch_positions)
            for idx in parent.vision_layer_indices
        ]

        fusion = parent.post_merger_fusion
        fdtype = next(fusion.parameters()).dtype
        h_out = fusion(
            h_base.to(fdtype),
            [h.to(fdtype) for h in h_layers],
            parent._task_query,
            parent._fail_query,
        )
        if parent.post_merger_adapter is not None:
            adapter = parent.post_merger_adapter
            h_out = h_out + adapter.gamma * adapter(h_out)
        return h_out.to(out_dtype)


class OV2RoutedMaTCA(nn.Module):
    """Frozen OneVision-2 + optional pre-LLM routing/MoE + post-LLM MaTCA head."""

    def __init__(
        self,
        model_id=DEFAULT_OV2_MODEL_ID,
        revision=DEFAULT_OV2_REVISION,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_pixels=200704,
        num_classifiers=3,
        target_layer_indices=None,
        pooling_mode="tcond",
        dropout_rate=0.1,
        use_hier_fusion=False,
        use_tgif_fusion=False,
        use_router=False,
        use_fuse_then_route=False,
        use_hier_router=False,
        depth_fusion_mode="query_cond",
        router_mode="contrastive",
        gate_type="sigmoid",
        gate_style="multiplicative",
        share_query=False,
        fusion_mode="static",
        vision_layer_indices=None,
        router_dim=256,
        use_moe=False,
        num_experts=4,
        moe_top_k=2,
        load_balance_coef=0.01,
        moe_gate_supervision=False,
        use_merger_adapter=False,
        merger_adapter_rank=64,
        use_nested_guided_fusion=False,
        ngf_layer_weight_mode="text",
        ngf_inner_guiding=True,
        nested_replace_base=True,
        use_ngf_sequential=False,
        ngf_inner_beta_l2=0.0,
        layer_balance_coef=0.0,
        ngf_tap="block",
        ngf_intermediate_only=False,
        ngf_full_connector=False,
        use_post_merger_alf=False,
        post_merger_adapter=True,
        post_merger_adapter_rank=64,
        alf_router_dim=256,
    ):
        super().__init__()

        if use_tgif_fusion:
            if use_router or use_hier_router or use_fuse_then_route or use_moe:
                raise ValueError(
                    "--use_tgif_fusion is mutually exclusive with "
                    "--use_router, --use_hier_router, --use_fuse_then_route, and --use_moe"
                )
            use_hier_fusion = True

        if use_fuse_then_route:
            if use_hier_router or use_moe:
                raise ValueError(
                    "--use_fuse_then_route is mutually exclusive with "
                    "--use_hier_router and --use_moe"
                )
            use_hier_fusion = True
            use_router = True

        if use_hier_router and (use_hier_fusion or use_router or use_moe):
            raise ValueError(
                "--use_hier_router is mutually exclusive with "
                "--use_hier_fusion, --use_router, --use_fuse_then_route, and --use_moe"
            )

        if use_nested_guided_fusion and (
            use_router
            or use_hier_router
            or use_fuse_then_route
            or use_tgif_fusion
            or use_hier_fusion
            or use_moe
        ):
            raise ValueError(
                "--use_nested_guided_fusion is mutually exclusive with "
                "--use_router, --use_hier_router, --use_fuse_then_route, "
                "--use_tgif_fusion, --use_hier_fusion, and --use_moe"
            )

        if ngf_tap not in ("block", "ffn_act"):
            raise ValueError(f"ngf_tap must be 'block' or 'ffn_act', got {ngf_tap!r}")
        if ngf_tap == "ffn_act" and not use_nested_guided_fusion:
            raise ValueError("--ngf_tap ffn_act requires --use_nested_guided_fusion")
        if ngf_intermediate_only and not use_nested_guided_fusion:
            raise ValueError(
                "--ngf_intermediate_only requires --use_nested_guided_fusion"
            )
        if ngf_full_connector:
            if not use_nested_guided_fusion:
                raise ValueError(
                    "--ngf_full_connector requires --use_nested_guided_fusion"
                )
            if use_merger_adapter:
                raise ValueError(
                    "--ngf_full_connector is mutually exclusive with "
                    "--use_merger_adapter (the connector supersedes it)"
                )

        if use_post_merger_alf:
            if (
                use_router
                or use_hier_router
                or use_fuse_then_route
                or use_tgif_fusion
                or use_hier_fusion
                or use_moe
                or use_nested_guided_fusion
            ):
                raise ValueError(
                    "--use_post_merger_alf is mutually exclusive with "
                    "--use_router, --use_hier_router, --use_fuse_then_route, "
                    "--use_tgif_fusion, --use_hier_fusion, --use_moe, and "
                    "--use_nested_guided_fusion"
                )
            if use_merger_adapter:
                raise ValueError(
                    "--use_post_merger_alf uses --post_merger_adapter (4096-d); "
                    "disable the pre-merger --use_merger_adapter (1024-d)"
                )

        self.device = torch.device(device)
        self.num_classifiers = num_classifiers
        self.pooling_mode = pooling_mode
        self.dropout_rate = dropout_rate
        self.use_fuse_then_route = use_fuse_then_route
        self.use_tgif_fusion = use_tgif_fusion
        self.use_hier_fusion = use_hier_fusion
        self.use_router = use_router
        self.use_hier_router = use_hier_router
        self.depth_fusion_mode = depth_fusion_mode
        self.router_mode = router_mode
        self.gate_type = gate_type
        self.gate_style = gate_style
        self.fusion_mode = fusion_mode
        self.model_id = model_id
        self.revision = revision
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.load_balance_coef = load_balance_coef
        self.moe_gate_supervision = moe_gate_supervision
        self.use_merger_adapter = use_merger_adapter
        self.merger_adapter_rank = merger_adapter_rank
        self.use_nested_guided_fusion = use_nested_guided_fusion
        self.ngf_layer_weight_mode = ngf_layer_weight_mode
        self.ngf_inner_guiding = ngf_inner_guiding
        self.nested_replace_base = nested_replace_base
        self.use_ngf_sequential = use_ngf_sequential
        self.ngf_inner_beta_l2 = ngf_inner_beta_l2
        self.layer_balance_coef = layer_balance_coef
        self.ngf_tap = ngf_tap
        self.ngf_intermediate_only = ngf_intermediate_only
        self.ngf_full_connector = ngf_full_connector
        self.use_post_merger_alf = use_post_merger_alf
        self.post_merger_adapter_enabled = post_merger_adapter
        self.post_merger_adapter_rank = post_merger_adapter_rank
        self.alf_router_dim = alf_router_dim

        # ----- Load and freeze the OneVision-2 backbone -----
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )

        # Bound the visual token budget (matches the OV2 baseline default).
        if max_pixels is not None:
            self.processor.image_processor.max_pixels = int(max_pixels)
            if hasattr(self.processor.image_processor, "size"):
                self.processor.image_processor.size["longest_edge"] = int(max_pixels)

        model_kwargs = {
            "revision": revision,
            "trust_remote_code": True,
            "dtype": dtype,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Single device (no device_map sharding) so gradients can flow cleanly
        # through the frozen LM to the upstream router during training.
        self.vlm = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.vlm.to(self.device)
        self.vlm.eval()
        for param in self.vlm.parameters():
            param.requires_grad = False

        # ----- Resolve dimensions and the vision submodules -----
        config = self.vlm.config
        vision_config = config.vision_config
        text_config = config.text_config

        self.vision_dim = vision_config.hidden_size          # 1024
        self.feature_dim = text_config.hidden_size           # 4096 (LM hidden)
        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)

        self.vision_model = self.vlm.model.visual

        if target_layer_indices is None:
            target_layer_indices = [19, 28, 36]
        if len(target_layer_indices) != self.num_classifiers:
            raise ValueError(
                f"target_layer_indices has {len(target_layer_indices)} entries "
                f"but num_classifiers={self.num_classifiers}"
            )
        self.target_layer_indices = target_layer_indices

        if vision_layer_indices is None:
            vision_layer_indices = [9, 17, 24]
        self.vision_layer_indices = vision_layer_indices
        if (
            self.use_hier_fusion
            or self.use_hier_router
            or self.use_nested_guided_fusion
            or self.use_post_merger_alf
        ):
            self._validate_vision_layer_indices()

        # ----- Build the (trainable) post-LLM MaTCA head -----
        self.att_poolings = nn.ModuleList(
            TaskConditionedPooling(input_dim=self.feature_dim, dropout=dropout_rate)
            for _ in range(self.num_classifiers)
        )
        self.hybrid_poolings = nn.ModuleList(
            HybridAttentionPooling(input_dim=self.feature_dim, dropout=dropout_rate)
            for _ in range(self.num_classifiers)
        )
        self.classifiers = nn.ModuleList(
            nn.Sequential(
                MLP_BLOCK(self.feature_dim, 1024, dropout_rate),
                MLP_BLOCK(1024, 256, dropout_rate),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1),
            )
            for _ in range(self.num_classifiers)
        )
        if fusion_mode == "dynamic":
            self.layer_fusion = DynamicLayerFusion(self.num_classifiers, self.feature_dim)
        else:
            self.layer_fusion = LearnedLayerFusion(self.num_classifiers)
        self.fused_classifier = nn.Sequential(
            MLP_BLOCK(self.feature_dim, 1024, dropout_rate),
            MLP_BLOCK(1024, 256, dropout_rate),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

        # ----- Build the (trainable) upstream Stage-1 modules -----
        self.hier_fusion = None
        if self.use_hier_fusion:
            num_fusion_layers = len(self.vision_layer_indices) + (
                1 if self.use_tgif_fusion else 0
            )
            _fusion = fusion_mode if fusion_mode in ("static", "dynamic", "mean", "text") else "static"
            self.hier_fusion = HierarchicalVisionFusion(
                vision_dim=self.vision_dim,
                num_layers=num_fusion_layers,
                fusion_mode=_fusion,
                query_dim=self.feature_dim if _fusion == "text" else None,
            )

        # The dual-query router is always built when routing OR MoE is enabled:
        # the MoE reuses the router's gated evidence streams (build_streams).
        self.router = None
        self.hier_router = None
        self.moe = None
        if self.use_hier_router:
            num_depths = len(self.vision_layer_indices) + 1
            self.hier_router = HierarchicalDualQueryRouter(
                vision_dim=self.vision_dim,
                query_dim=self.feature_dim,
                num_depths=num_depths,
                router_dim=router_dim,
                router_mode=router_mode,
                gate_type=gate_type,
                gate_style=gate_style,
                share_query=share_query,
                depth_fusion_mode=depth_fusion_mode,
            )
        elif self.use_router or self.use_moe:
            self.router = DualQueryRouter(
                vision_dim=self.vision_dim,
                query_dim=self.feature_dim,
                router_dim=router_dim,
                router_mode=router_mode,
                gate_type=gate_type,
                gate_style=gate_style,
                share_query=share_query,
            )
        self.nested_fusion = None
        if self.use_nested_guided_fusion:
            # Intermediate-only fusion drops the base (V_base) depth entirely.
            num_depths = len(self.vision_layer_indices)
            if not self.ngf_intermediate_only:
                num_depths += 1  # + base merger input
            if self.use_ngf_sequential:
                self.nested_fusion = SequentialNestedFusion(
                    vision_dim=self.vision_dim,
                    query_dim=self.feature_dim,
                    num_layers=num_depths,
                    router_dim=router_dim,
                    router_mode=router_mode,
                    gate_type=gate_type,
                    gate_style=gate_style,
                    share_query=share_query,
                )
            else:
                self.nested_fusion = NestedGuidedFusion(
                    vision_dim=self.vision_dim,
                    query_dim=self.feature_dim,
                    num_layers=num_depths,
                    router_dim=router_dim,
                    router_mode=router_mode,
                    gate_type=gate_type,
                    gate_style=gate_style,
                    share_query=share_query,
                    layer_weight_mode=ngf_layer_weight_mode,
                    layer_balance_coef=layer_balance_coef,
                    inner_guiding=ngf_inner_guiding,
                    replace_base=nested_replace_base,
                )

        # FFN-Act taps live at the FFN expansion width (intermediate_size); a
        # per-depth linear projects them back to vision_dim before fusion.
        self.ffn_act_down = None
        if self.use_nested_guided_fusion and self.ngf_tap == "ffn_act":
            intermediate_size = self.vlm.config.vision_config.intermediate_size
            self.ffn_act_down = nn.ModuleList(
                [
                    nn.Linear(intermediate_size, self.vision_dim)
                    for _ in self.vision_layer_indices
                ]
            )
            self.ffn_act_down.to(self.device)

        if self.use_moe:
            self.moe = MoEFailureRouter(
                vision_dim=self.vision_dim,
                query_dim=self.feature_dim,
                num_streams=self.router.num_streams,
                num_experts=num_experts,
                top_k=moe_top_k,
                router_dim=router_dim,
                load_balance_coef=load_balance_coef,
            )

        # Move all trainable submodules to the working device/precision (float32).
        self.att_poolings.to(self.device)
        self.hybrid_poolings.to(self.device)
        self.classifiers.to(self.device)
        self.layer_fusion.to(self.device)
        self.fused_classifier.to(self.device)
        if self.hier_fusion is not None:
            self.hier_fusion.to(self.device)
        if self.router is not None:
            self.router.to(self.device)
        if self.hier_router is not None:
            self.hier_router.to(self.device)
        if self.moe is not None:
            self.moe.to(self.device)
        if self.nested_fusion is not None:
            self.nested_fusion.to(self.device)

        self.merger_adapter = None
        if self.use_merger_adapter:
            native_merger = self.vision_model.merger
            for param in native_merger.parameters():
                param.requires_grad = False
            output_dim = native_merger.mlp[-1].out_features
            self.merger_adapter = MergerAdapter(
                context_dim=self.vision_dim,
                hidden_size=native_merger.hidden_size,
                output_dim=output_dim,
                rank=merger_adapter_rank,
            )
            self.merger_adapter.to(self.device)

        # Full trainable connector: a warm-started, fully-trainable clone of the
        # native patch merger that REPLACES it for the NGF path (no frozen merger,
        # no rank-64 parallel adapter). Built in float32 like the other trainable
        # Stage-1 modules; the wrapper casts to/from the backbone dtype.
        self.full_connector = None
        if self.use_nested_guided_fusion and self.ngf_full_connector:
            native_merger = self.vision_model.merger
            for param in native_merger.parameters():
                param.requires_grad = False
            self.full_connector = copy.deepcopy(native_merger).float()
            for param in self.full_connector.parameters():
                param.requires_grad = True
            self.full_connector.to(self.device)

        # Post-merger ALF: cross-attention over per-depth (frozen) merger outputs
        # in 4096-d LLM token space, plus an optional low-rank output adapter.
        self.post_merger_fusion = None
        self.post_merger_adapter = None
        if self.use_post_merger_alf:
            native_merger = self.vision_model.merger
            for param in native_merger.parameters():
                param.requires_grad = False
            self.post_merger_fusion = PostMergerCrossLayerFusion(
                feature_dim=self.feature_dim,
                query_dim=self.feature_dim,
                num_layers=len(self.vision_layer_indices),
                router_dim=alf_router_dim,
            )
            self.post_merger_fusion.to(self.device)
            if self.post_merger_adapter_enabled:
                self.post_merger_adapter = PostMergerOutputAdapter(
                    feature_dim=self.feature_dim,
                    rank=post_merger_adapter_rank,
                )
                self.post_merger_adapter.to(self.device)

        # ----- Install the upstream insertion hook -----
        self._encoder_hidden_states = None
        self._ffn_act_cache = {}
        self._task_query = None
        self._fail_query = None
        self._upstream_active = False
        self._last_gates = None
        self._last_depth_weights = None
        self._last_vision_fusion_weights = None
        self._last_layer_weights = None
        self._last_expert_usage = None
        self._aux_loss = None

        self._install_hooks()

        # Backprop through a frozen 8B LM is memory-heavy; checkpointing makes a
        # single-GPU upstream training run feasible.
        if (
            self.use_router
            or self.use_hier_fusion
            or self.use_hier_router
            or self.use_moe
            or self.use_merger_adapter
            or self.use_nested_guided_fusion
            or self.use_post_merger_alf
        ):
            try:
                self.vlm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except Exception as exc:
                print(f"[warn] could not enable gradient checkpointing: {exc}")

    def _validate_vision_layer_indices(self):
        """``vision_layer_indices`` index the encoder ``hidden_states`` tuple.

        OV2's vision encoder (24 blocks) returns ``num_hidden_layers + 1`` states:
        index 0 = post-embedding, index ``k`` = after block ``k - 1``, index 24 =
        after the final block (same tensor as ``hidden_states[-1]`` before post-norm).
        """
        num_blocks = self.vlm.config.vision_config.num_hidden_layers
        max_idx = num_blocks  # inclusive upper bound for hidden_states indexing
        for idx in self.vision_layer_indices:
            if idx < 0 or idx > max_idx:
                raise RuntimeError(
                    f"vision_layer_indices requested {idx} but the vision encoder "
                    f"hidden_states tuple has valid indices 0..{max_idx} "
                    f"({num_blocks} transformer blocks)."
                )

    # ---- Hook installation ----
    def _install_hooks(self):
        def encoder_hook(module, inputs, output):
            self._encoder_hidden_states = getattr(output, "hidden_states", None)

        self.vision_model.encoder.register_forward_hook(encoder_hook)

        # FFN-Act taps: per-layer forward hooks on the FFN post-GELU activation.
        # vision_layer_indices are treated as encoder BLOCK indices here (0-based,
        # 0..num_blocks-1), unlike the block-tap path where they index the
        # hidden_states tuple.
        if self.use_nested_guided_fusion and self.ngf_tap == "ffn_act":
            encoder_layers = self.vision_model.encoder.layers
            num_blocks = len(encoder_layers)

            def make_ffn_hook(block_idx):
                def hook(_module, _inputs, output):
                    # The ViT is frozen, so the activation is a constant feature
                    # input; detach it. The trainable entry point is the per-layer
                    # down-projection (its weights carry the gradient), so the act
                    # itself need not be on the autograd graph.
                    self._ffn_act_cache[block_idx] = output.detach()
                return hook

            for idx in self.vision_layer_indices:
                if idx < 0 or idx >= num_blocks:
                    raise RuntimeError(
                        f"--ngf_tap ffn_act treats vision_layer_indices as encoder "
                        f"block indices (0..{num_blocks - 1}); got {idx}."
                    )
                encoder_layers[idx].mlp.activation_fn.register_forward_hook(
                    make_ffn_hook(idx)
                )

        native_merger = self.vision_model.merger
        if self.use_post_merger_alf:
            self.vision_model.merger = _PostMergerALFMerger(native_merger, self)
        elif self.ngf_full_connector:
            self.vision_model.merger = _ReplacingRoutedMerger(
                self.full_connector, self
            )
        elif self.use_merger_adapter:
            self.vision_model.merger = _AdaptedRoutedMerger(
                native_merger, self, self.merger_adapter
            )
        else:
            self.vision_model.merger = _RoutedMerger(native_merger, self)

    # ---- Upstream application (called from inside the vision forward) ----
    def apply_upstream(self, x):
        """Apply hierarchical fusion + routing/MoE to pre-merger visual tokens.

        ``x`` is the configured vision layer in the vision hidden dim, shaped
        ``[1, total_patches, vision_dim]``. Computation is done in float32 for
        stability and cast back to the backbone dtype on return.
        """
        if not self._upstream_active:
            return x

        orig_dtype = x.dtype
        x32 = x.float()

        if self.use_nested_guided_fusion:
            if self.ngf_tap == "ffn_act":
                missing = [
                    idx for idx in self.vision_layer_indices
                    if idx not in self._ffn_act_cache
                ]
                if missing:
                    raise RuntimeError(
                        "FFN-Act taps were not captured for encoder blocks "
                        f"{missing}; the FFN-Act hooks did not fire."
                    )
                # The frozen-ViT activations are constant features; the per-layer
                # down-projection (trainable) maps them to vision_dim and carries
                # the gradient for the fusion + connector downstream.
                layer_tokens = [
                    self.ffn_act_down[k](self._ffn_act_cache[idx].float())
                    for k, idx in enumerate(self.vision_layer_indices)
                ]
            else:
                if self._encoder_hidden_states is None:
                    raise RuntimeError(
                        "Nested guided fusion is enabled but encoder hidden states "
                        "were not captured. The encoder hook did not fire."
                    )
                hidden_states = self._encoder_hidden_states
                for idx in self.vision_layer_indices:
                    if idx >= len(hidden_states):
                        raise RuntimeError(
                            f"vision_layer_indices requested {idx} but the encoder "
                            f"returned only {len(hidden_states)} hidden states."
                        )
                layer_tokens = [
                    hidden_states[idx].float() for idx in self.vision_layer_indices
                ]

            if not self.ngf_intermediate_only:
                layer_tokens.append(x32)  # base = merger input, last depth
            x32 = self.nested_fusion(
                layer_tokens, x32, self._task_query, self._fail_query
            )
            self._last_vision_fusion_weights = self.nested_fusion.last_alpha
            self._aux_loss = self.nested_fusion.last_aux_loss
            return x32.to(orig_dtype)

        if self.use_hier_router:
            if self._encoder_hidden_states is None:
                raise RuntimeError(
                    "Hierarchical router is enabled but encoder hidden states were "
                    "not captured. The encoder hook did not fire."
                )
            hidden_states = self._encoder_hidden_states
            for idx in self.vision_layer_indices:
                if idx >= len(hidden_states):
                    raise RuntimeError(
                        f"vision_layer_indices requested {idx} but the encoder "
                        f"returned only {len(hidden_states)} hidden states."
                    )
            layer_tokens = [
                hidden_states[idx].float() for idx in self.vision_layer_indices
            ]
            layer_tokens.append(x32)
            x32, gates = self.hier_router(
                x32, layer_tokens, self._task_query, self._fail_query
            )
            self._last_gates = gates
            self._last_depth_weights = gates.get("depth_weights")
        elif self.use_hier_fusion:
            if self._encoder_hidden_states is None:
                raise RuntimeError(
                    "Hierarchical fusion is enabled but encoder hidden states were "
                    "not captured. The encoder hook did not fire."
                )
            hidden_states = self._encoder_hidden_states
            for idx in self.vision_layer_indices:
                if idx >= len(hidden_states):
                    raise RuntimeError(
                        f"vision_layer_indices requested {idx} but the encoder "
                        f"returned only {len(hidden_states)} hidden states."
                    )
            layers = [hidden_states[idx].float() for idx in self.vision_layer_indices]
            if self.use_tgif_fusion:
                layers.append(x32)
            if self.fusion_mode == "text":
                weights = self.hier_fusion._depth_weights(
                    layers, self._task_query, self._fail_query
                )
                self._last_vision_fusion_weights = weights.detach().mean(dim=0)
                x32 = self.hier_fusion(
                    x32, layers, self._task_query, self._fail_query
                )
            else:
                self._last_vision_fusion_weights = self.hier_fusion._depth_weights(
                    layers
                ).detach()
                x32 = self.hier_fusion(x32, layers)

        if self.use_moe:
            # Reuse the router's gated, purely-visual evidence streams, then route
            # through the failure-expert mixture (grounding invariant preserved).
            self.router._t_task = self._task_query
            self.router._t_fail = self._fail_query
            streams, gates = self.router.build_streams(x32)
            x32, moe_gates = self.moe(x32, streams, self._task_query, self._fail_query)
            self._last_gates = gates
            self._last_expert_usage = moe_gates["expert_usage"]
            self._aux_loss = self.moe.last_aux_loss
        elif self.use_router:
            x32, gates = self.router(x32, self._task_query, self._fail_query)
            self._last_gates = gates

        return x32.to(orig_dtype)

    # ---- Query construction (task + failure template) ----
    def _compute_query_embeddings(self, tasks):
        task_text = tasks[0] if isinstance(tasks, (list, tuple)) else tasks
        fail_text = (
            "Visual evidence that the following robot task was not successfully "
            f"completed: {task_text}"
        )

        tokenizer = self.processor.tokenizer
        embed = self.vlm.get_input_embeddings()

        def pooled_embedding(text):
            ids = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(self.device)
            with torch.no_grad():
                embeddings = embed(ids)
            return embeddings.float().mean(dim=1)

        self._task_query = pooled_embedding(task_text)
        self._fail_query = pooled_embedding(fail_text)

    # ---- Input preparation (mirror of the OV2 baseline batch builder) ----
    def _prepare_inputs(self, images_batch, tasks, prompt_styles=None):
        if prompt_styles is None:
            prompt_styles = [None] * len(tasks)

        prompts = []
        flat_images = []
        for images, task, prompt_style in zip(images_batch, tasks, prompt_styles):
            if not isinstance(images, list):
                images = [images]
            messages = build_messages(images=images, task=task, prompt_style=prompt_style)
            prompts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            flat_images.extend(images)

        inputs = self.processor(
            text=prompts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )

        prepared = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared

    def _make_text_mask(self, input_ids, attention_mask):
        text_mask = attention_mask.bool()
        if self.image_token_id is not None:
            text_mask = text_mask & (input_ids != self.image_token_id)
        if self.video_token_id is not None:
            text_mask = text_mask & (input_ids != self.video_token_id)
        return text_mask

    def _pool(self, features, layer_idx, text_mask, attention_mask):
        if self.pooling_mode == "tcond":
            pooled, _, _ = self.att_poolings[layer_idx](
                features, text_mask=text_mask, attention_mask=attention_mask
            )
            return pooled
        if self.pooling_mode == "hybrid":
            pooled, _ = self.hybrid_poolings[layer_idx](
                features, attention_mask=attention_mask
            )
            return pooled
        if self.pooling_mode == "last_token":
            last_indices = attention_mask.long().sum(dim=1) - 1
            batch_indices = torch.arange(features.shape[0], device=features.device)
            return features[batch_indices, last_indices, :]
        if self.pooling_mode == "text_mean":
            mask = text_mask.unsqueeze(-1).to(features.dtype)
            return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

    # ---- Forward ----
    def forward(self, images, tasks, prompt_styles=None):
        self._upstream_active = (
            self.use_hier_fusion
            or self.use_router
            or self.use_hier_router
            or self.use_moe
            or self.use_merger_adapter
            or self.use_nested_guided_fusion
            or self.use_post_merger_alf
        )
        self._aux_loss = None

        if self._upstream_active and len(tasks) != 1:
            raise ValueError(
                "Routing/hierarchical fusion/MoE is implemented for batch_size=1 "
                f"(got {len(tasks)} samples). Use batch_size=1 for routed runs."
            )

        if (
            self.use_router
            or self.use_moe
            or self.use_hier_router
            or self.use_nested_guided_fusion
            or self.use_post_merger_alf
            or (self.use_hier_fusion and self.fusion_mode == "text")
        ):
            self._compute_query_embeddings(tasks)

        inputs = self._prepare_inputs(images, tasks, prompt_styles)

        grad_path = self._upstream_active and self.training

        if grad_path:
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            features = [hidden_states[idx].to(torch.float32) for idx in self.target_layer_indices]
        else:
            with torch.inference_mode():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            hidden_states = outputs.hidden_states
            features = [
                hidden_states[idx].detach().to(torch.float32)
                for idx in self.target_layer_indices
            ]

        attention_mask = inputs["attention_mask"].to(self.device)
        text_mask = self._make_text_mask(inputs["input_ids"], attention_mask).to(self.device)

        pooled_layers = []
        logits = []
        for i in range(self.num_classifiers):
            pooled = self._pool(features[i], i, text_mask, attention_mask)
            logits.append(self.classifiers[i](pooled))
            pooled_layers.append(pooled)

        stacked = torch.stack(pooled_layers, dim=1)
        fused, layer_weights = self.layer_fusion(stacked)
        self._last_layer_weights = layer_weights.detach()
        logits.append(self.fused_classifier(fused))

        return logits

    @torch.no_grad()
    def predict(self, images, tasks, prompt_styles=None, prediction_mode="fusion", voting=False):
        # ``voting`` is accepted for signature compatibility with the Qwen model;
        # the head-voting behavior is selected via ``prediction_mode`` instead.
        self.eval()
        logits = self.forward(images, tasks, prompt_styles=prompt_styles)

        if prediction_mode == "fusion":
            prob = torch.sigmoid(logits[-1].squeeze(-1))
            return (prob > 0.5).float(), prob

        head_logits = logits[: self.num_classifiers]
        head_probs = torch.stack(
            [torch.sigmoid(logit.squeeze(-1)) for logit in head_logits], dim=0
        )

        if prediction_mode == "head_average":
            avg = head_probs.mean(dim=0)
            return (avg > 0.5).float(), avg
        if prediction_mode == "head_majority":
            votes = (head_probs > 0.5).float().sum(dim=0)
            threshold = (self.num_classifiers // 2) + 1
            avg = votes / self.num_classifiers
            return (votes >= threshold).float(), avg

        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    # ---- Trainable parameter collection ----
    def trainable_parameters(self):
        params = []
        for i in range(self.num_classifiers):
            params.extend(self.classifiers[i].parameters())
            if self.pooling_mode == "tcond":
                params.extend(self.att_poolings[i].parameters())
            if self.pooling_mode == "hybrid":
                params.extend(self.hybrid_poolings[i].parameters())
        params.extend(self.layer_fusion.parameters())
        params.extend(self.fused_classifier.parameters())
        if self.hier_fusion is not None:
            params.extend(self.hier_fusion.parameters())
        if self.router is not None:
            params.extend(self.router.parameters())
        if self.hier_router is not None:
            params.extend(self.hier_router.parameters())
        if self.moe is not None:
            params.extend(self.moe.parameters())
        if self.nested_fusion is not None:
            params.extend(self.nested_fusion.parameters())
        if self.merger_adapter is not None:
            params.extend(self.merger_adapter.parameters())
        if self.ffn_act_down is not None:
            params.extend(self.ffn_act_down.parameters())
        if self.full_connector is not None:
            params.extend(self.full_connector.parameters())
        if self.post_merger_fusion is not None:
            params.extend(self.post_merger_fusion.parameters())
        if self.post_merger_adapter is not None:
            params.extend(self.post_merger_adapter.parameters())
        return params

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    # ---- Checkpointing ----
    def save_classifier(self, path="./checkpoints", epoch=None):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "num_classifiers": self.num_classifiers,
            "dropout_rate": self.dropout_rate,
            "target_layer_indices": self.target_layer_indices,
            "vision_layer_indices": self.vision_layer_indices,
            "pooling_mode": self.pooling_mode,
            "fusion_mode": self.fusion_mode,
            "use_hier_fusion": self.use_hier_fusion,
            "use_tgif_fusion": self.use_tgif_fusion,
            "use_router": self.use_router,
            "use_fuse_then_route": self.use_fuse_then_route,
            "use_hier_router": self.use_hier_router,
            "depth_fusion_mode": self.depth_fusion_mode,
            "router_mode": self.router_mode,
            "gate_type": self.gate_type,
            "gate_style": self.gate_style,
            "use_moe": self.use_moe,
            "num_experts": self.num_experts,
            "moe_top_k": self.moe_top_k,
            "load_balance_coef": self.load_balance_coef,
            "use_merger_adapter": self.use_merger_adapter,
            "merger_adapter_rank": self.merger_adapter_rank,
            "use_nested_guided_fusion": self.use_nested_guided_fusion,
            "ngf_layer_weight_mode": self.ngf_layer_weight_mode,
            "ngf_inner_guiding": self.ngf_inner_guiding,
            "nested_replace_base": self.nested_replace_base,
            "use_ngf_sequential": self.use_ngf_sequential,
            "ngf_inner_beta_l2": self.ngf_inner_beta_l2,
            "layer_balance_coef": self.layer_balance_coef,
            "ngf_tap": self.ngf_tap,
            "ngf_intermediate_only": self.ngf_intermediate_only,
            "ngf_full_connector": self.ngf_full_connector,
            "use_post_merger_alf": self.use_post_merger_alf,
            "post_merger_adapter": self.post_merger_adapter_enabled,
            "post_merger_adapter_rank": self.post_merger_adapter_rank,
            "alf_router_dim": self.alf_router_dim,
            "layer_fusion": self.layer_fusion.state_dict(),
            "fused_classifier": self.fused_classifier.state_dict(),
        }
        for i in range(self.num_classifiers):
            checkpoint[f"classifier_{i}"] = self.classifiers[i].state_dict()
            checkpoint[f"attention_pooling_{i}"] = self.att_poolings[i].state_dict()
            checkpoint[f"hybrid_pooling_{i}"] = self.hybrid_poolings[i].state_dict()
        if self.hier_fusion is not None:
            checkpoint["hier_fusion"] = self.hier_fusion.state_dict()
        if self.router is not None:
            checkpoint["router"] = self.router.state_dict()
        if self.hier_router is not None:
            checkpoint["hier_router"] = self.hier_router.state_dict()
        if self.moe is not None:
            checkpoint["moe"] = self.moe.state_dict()
        if self.nested_fusion is not None:
            checkpoint["nested_fusion"] = self.nested_fusion.state_dict()
        if self.merger_adapter is not None:
            checkpoint["merger_adapter"] = self.merger_adapter.state_dict()
        if self.ffn_act_down is not None:
            checkpoint["ffn_act_down"] = self.ffn_act_down.state_dict()
        if self.full_connector is not None:
            checkpoint["full_connector"] = self.full_connector.state_dict()
        if self.post_merger_fusion is not None:
            checkpoint["post_merger_fusion"] = self.post_merger_fusion.state_dict()
        if self.post_merger_adapter is not None:
            checkpoint["post_merger_adapter"] = self.post_merger_adapter.state_dict()

        filename = "components.pt" if epoch is None else f"components_epoch_{epoch}.pt"
        if epoch is not None:
            checkpoint["epoch"] = epoch
        full_path = os.path.join(path, filename)
        torch.save(checkpoint, full_path)
        print(f"Saved routed MaTCA components to {full_path}")

    def load_classifier(self, path, strict=True):
        assert os.path.isfile(path), f"Checkpoint not found: {path}"
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        for i in range(self.num_classifiers):
            self.classifiers[i].load_state_dict(checkpoint[f"classifier_{i}"], strict=strict)
            self.att_poolings[i].load_state_dict(checkpoint[f"attention_pooling_{i}"], strict=strict)
            if f"hybrid_pooling_{i}" in checkpoint:
                self.hybrid_poolings[i].load_state_dict(
                    checkpoint[f"hybrid_pooling_{i}"], strict=strict
                )
        self.layer_fusion.load_state_dict(checkpoint["layer_fusion"], strict=strict)
        self.fused_classifier.load_state_dict(checkpoint["fused_classifier"], strict=strict)

        if self.hier_fusion is not None and "hier_fusion" in checkpoint:
            self.hier_fusion.load_state_dict(checkpoint["hier_fusion"], strict=strict)
        if self.router is not None and "router" in checkpoint:
            self.router.load_state_dict(checkpoint["router"], strict=strict)
        if self.hier_router is not None and "hier_router" in checkpoint:
            self.hier_router.load_state_dict(checkpoint["hier_router"], strict=strict)
        if self.moe is not None and "moe" in checkpoint:
            self.moe.load_state_dict(checkpoint["moe"], strict=strict)
        if self.nested_fusion is not None and "nested_fusion" in checkpoint:
            self.nested_fusion.load_state_dict(checkpoint["nested_fusion"], strict=strict)
        if self.merger_adapter is not None and "merger_adapter" in checkpoint:
            self.merger_adapter.load_state_dict(checkpoint["merger_adapter"], strict=strict)
        if self.ffn_act_down is not None and "ffn_act_down" in checkpoint:
            self.ffn_act_down.load_state_dict(checkpoint["ffn_act_down"], strict=strict)
        if self.full_connector is not None and "full_connector" in checkpoint:
            self.full_connector.load_state_dict(checkpoint["full_connector"], strict=strict)
        if self.post_merger_fusion is not None and "post_merger_fusion" in checkpoint:
            self.post_merger_fusion.load_state_dict(
                checkpoint["post_merger_fusion"], strict=strict
            )
        if self.post_merger_adapter is not None and "post_merger_adapter" in checkpoint:
            self.post_merger_adapter.load_state_dict(
                checkpoint["post_merger_adapter"], strict=strict
            )

        return checkpoint.get("epoch")

    def cleanup(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training / validation utilities
# ---------------------------------------------------------------------------
def validate_model(model, val_dataset, batch_size, prediction_mode="fusion"):
    model.eval()
    correct = 0
    total = 0
    eval_batch_size = 1 if (
        model.use_router
        or model.use_hier_fusion
        or model.use_hier_router
        or model.use_moe
        or model.use_merger_adapter
        or model.use_nested_guided_fusion
        or model.use_post_merger_alf
    ) else batch_size

    with torch.no_grad():
        for start in range(0, len(val_dataset), eval_batch_size):
            end = min(start + eval_batch_size, len(val_dataset))
            entries = val_dataset[start:end]

            tasks = entries["task"]
            images = entries["images"]
            prompt_styles = entries.get("prompt_style", [None] * len(tasks))
            labels = torch.tensor(
                [label_to_binary(label) for label in entries["label"]],
                dtype=torch.float32,
                device=model.device,
            )

            try:
                predictions, _ = model.predict(
                    images, tasks, prompt_styles=prompt_styles, prediction_mode=prediction_mode
                )
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            except Exception as exc:
                print(f"[warn] skipping validation batch: {exc}")
                continue

    return correct / total if total > 0 else 0.0


def train_model(model, train_dataset, val_dataset, config):
    """Train the head (and upstream/MoE modules) on top of the frozen backbone."""
    criterion = nn.BCEWithLogitsLoss()

    trainable_params = model.trainable_parameters()
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    batch_size = 1 if (
        model.use_router
        or model.use_hier_fusion
        or model.use_hier_router
        or model.use_moe
        or model.use_merger_adapter
        or model.use_nested_guided_fusion
        or model.use_post_merger_alf
    ) else config["batch_size"]
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"] * steps_per_epoch,
        eta_min=config["lr"] * 0.01,
    )

    loss_mode = config.get("loss_mode", "fusion")
    prediction_mode = config.get("prediction_mode", "fusion")
    best_val_acc = 0.0

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        model.train()
        model.vlm.eval()

        running_loss = 0.0
        running_aux = 0.0
        num_batches = 0
        correct = 0
        total = 0

        for start in range(0, len(train_dataset), batch_size):
            end = min(start + batch_size, len(train_dataset))
            if end - start < batch_size:
                continue
            entries = train_dataset[start:end]

            tasks = entries["task"]
            images = entries["images"]
            prompt_styles = entries.get("prompt_style", [None] * len(tasks))
            labels = torch.tensor(
                [label_to_binary(label) for label in entries["label"]],
                dtype=torch.float32,
                device=model.device,
            )

            optimizer.zero_grad()
            logits = model(images, tasks, prompt_styles=prompt_styles)

            if loss_mode == "fusion":
                selected = [logits[-1]]
            elif loss_mode == "per_head":
                selected = logits[: model.num_classifiers]
            elif loss_mode == "all":
                selected = logits
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")

            losses = [criterion(logit.squeeze(-1), labels) for logit in selected]
            loss = sum(losses) / len(losses)

            # Add the MoE auxiliary (load-balance) loss when present.
            aux_value = 0.0
            if model._aux_loss is not None:
                loss = loss + model._aux_loss
                aux_value = float(model._aux_loss.detach().item())

            # Optional L2 shrinkage on the NGF inner-residual scales (beta_l /
            # beta_seq). Dials back inner-guiding capacity toward the inner-OFF
            # baseline, which transfers better cross-dataset.
            beta_l2 = getattr(model, "ngf_inner_beta_l2", 0.0)
            if beta_l2 and model.nested_fusion is not None:
                betas = getattr(model.nested_fusion, "inner_betas", None)
                if betas is None:
                    betas = getattr(model.nested_fusion, "beta_seq", None)
                if betas is not None:
                    beta_pen = beta_l2 * betas.pow(2).sum()
                    loss = loss + beta_pen
                    aux_value += float(beta_pen.detach().item())

            # Optional supervised gate alignment when failure-mode ids are present.
            if model.moe is not None and model.moe_gate_supervision:
                fmode = entries.get("failure_mode_id", [None])[0] if hasattr(entries, "get") else None
                sup = model.moe.supervised_gate_loss(fmode)
                if sup is not None:
                    loss = loss + sup

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_aux += aux_value
            num_batches += 1
            with torch.no_grad():
                prob = torch.sigmoid(logits[-1].squeeze(-1))
                predictions = (prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            if num_batches % 50 == 0:
                acc = correct / total if total else 0.0
                print(
                    f"  step {num_batches}/{steps_per_epoch} "
                    f"loss={running_loss / num_batches:.4f} aux={running_aux / num_batches:.4f} "
                    f"acc={acc:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()

        val_acc = validate_model(model, val_dataset, batch_size, prediction_mode=prediction_mode)
        train_acc = correct / total if total else 0.0
        avg_loss = running_loss / num_batches if num_batches else 0.0
        print(
            f"  end of epoch {epoch + 1}: train_loss={avg_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if model.hier_fusion is not None:
            print(f"  alpha (hier residual) = {model.hier_fusion.alpha.item():.4f}")
            if model._last_vision_fusion_weights is not None:
                vw = model._last_vision_fusion_weights.detach().float().cpu().reshape(-1)
                vision_weights = [round(float(w), 3) for w in vw.tolist()]
                if model.use_tgif_fusion:
                    layer_labels = list(model.vision_layer_indices) + ["base"]
                    print(f"  TGIF fusion weights ({layer_labels}) = {vision_weights}")
                else:
                    print(f"  vision fusion weights (layers {model.vision_layer_indices}) = {vision_weights}")
        if model.router is not None:
            print(f"  beta  (router residual) = {model.router.beta.item():.4f}")
        if model.hier_router is not None:
            print(f"  beta  (hier router residual) = {model.hier_router.beta.item():.4f}")
            if model._last_depth_weights is not None:
                dw = model._last_depth_weights.detach().float().cpu().reshape(-1)
                weights = [round(float(w), 3) for w in dw.tolist()]
                print(f"  depth weights = {weights}")
        if model.moe is not None:
            print(f"  beta  (moe residual)    = {model.moe.beta.item():.4f}")
            if model._last_expert_usage is not None:
                usage = [round(float(u), 3) for u in model._last_expert_usage.tolist()]
                print(f"  expert usage = {usage}")
        if model.nested_fusion is not None:
            if model._last_vision_fusion_weights is not None:
                aw = model._last_vision_fusion_weights.detach().float().cpu().reshape(-1)
                layer_labels = list(model.vision_layer_indices) + ["base"]
                alpha_weights = [round(float(w), 3) for w in aw.tolist()]
                print(f"  NGF layer alpha ({layer_labels}) = {alpha_weights}")
                if max(alpha_weights) > 0.9:
                    print("  [collapse-guard] max(alpha) > 0.9 — consider raising --layer_balance_coef")
            if model.nested_fusion.last_gate_entropy is not None:
                print(f"  NGF mean patch gate entropy = {model.nested_fusion.last_gate_entropy:.4f}")
            layer_labels = list(model.vision_layer_indices) + ["base"]
            inner_betas = getattr(model.nested_fusion, "inner_betas", None)
            if inner_betas is not None:
                betas = [round(float(b), 4) for b in inner_betas.detach().float().cpu().tolist()]
                print(f"  NGF inner beta ({layer_labels}) = {betas}")
            beta_seq = getattr(model.nested_fusion, "beta_seq", None)
            if beta_seq is not None:
                bseq = [round(float(b), 4) for b in beta_seq.detach().float().cpu().tolist()]
                print(f"  NGF seq beta ({layer_labels}) = {bseq}")
        if model.merger_adapter is not None:
            print(f"  gamma (merger adapter)  = {model.merger_adapter.gamma.item():.4f}")
        if model.post_merger_fusion is not None:
            print(f"  beta  (post-merger ALF) = {model.post_merger_fusion.beta.item():.4f}")
            if model.post_merger_fusion.last_alpha is not None:
                aw = model.post_merger_fusion.last_alpha.detach().float().cpu().reshape(-1)
                alpha_weights = [round(float(w), 3) for w in aw.tolist()]
                print(
                    f"  ALF depth attn (layers {model.vision_layer_indices}) = {alpha_weights}"
                )
            if model.post_merger_adapter is not None:
                print(
                    f"  gamma (post-merger adapter) = "
                    f"{model.post_merger_adapter.gamma.item():.4f}"
                )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_classifier(path=config["save_path"], epoch=f"best_epoch_{epoch + 1}")
            print(f"  new best val_acc = {best_val_acc:.4f}")

        model.save_classifier(path=config["save_path"], epoch=epoch + 1)

    print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
    return best_val_acc
