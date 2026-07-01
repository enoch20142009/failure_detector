import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(42)


def label_to_binary(label):
    return 0 if label in ("0", "fail", 0, False) else 1


def build_guardian_execution_content(images, task, prompt_style):
    if not isinstance(images, list):
        images = [images]

    content = []

    if prompt_style == "single_viewpoint":
        if len(images) != 2:
            raise ValueError(f"single_viewpoint expects 2 images, got {len(images)}")

        content.extend(
            [
                {"type": "text", "text": "Image at the start of the subtask:"},
                {"type": "image", "image": images[0]},
                {"type": "text", "text": "\n\nImage at the end of the subtask:"},
                {"type": "image", "image": images[1]},
            ]
        )

    elif prompt_style == "three_viewpoints":
        if len(images) != 6:
            raise ValueError(f"three_viewpoints expects 6 images, got {len(images)}")

        view_names = ["Front view", "Left view", "Right view"]

        content.append(
            {"type": "text", "text": "Multiview images at the start of the subtask:\n"}
        )
        for view_name, image in zip(view_names, images[:3]):
            content.extend(
                [
                    {"type": "text", "text": f"{view_name}:\n"},
                    {"type": "image", "image": image},
                ]
            )

        content.append(
            {"type": "text", "text": "\nMultiview images at the end of the subtask:\n"}
        )
        for view_name, image in zip(view_names, images[3:]):
            content.extend(
                [
                    {"type": "text", "text": f"{view_name}:\n"},
                    {"type": "image", "image": image},
                ]
            )

    elif prompt_style == "four_viewpoints":
        if len(images) != 8:
            raise ValueError(f"four_viewpoints expects 8 images, got {len(images)}")

        view_names = ["Left view", "Right view", "Wrist view", "Front view"]

        content.append(
            {"type": "text", "text": "Multiview images at the start of the subtask:\n"}
        )
        for view_name, image in zip(view_names, images[:4]):
            content.extend(
                [
                    {"type": "text", "text": f"{view_name}:\n"},
                    {"type": "image", "image": image},
                ]
            )

        content.append(
            {"type": "text", "text": "\nMultiview images at the end of the subtask:\n"}
        )
        for view_name, image in zip(view_names, images[4:]):
            content.extend(
                [
                    {"type": "text", "text": f"{view_name}:\n"},
                    {"type": "image", "image": image},
                ]
            )

    else:
        raise ValueError(f"Unknown Guardian prompt_style: {prompt_style}")

    content.append(
        {
            "type": "text",
            "text": (
                "\nYou are a robot subtask execution failure detector. "
                f"Evaluate whether the robot successfully completed the subtask: {task}. "
                "Answer only success or fail."
            ),
        }
    )

    return content


def build_messages(images, task, prompt_style=None):
    if not isinstance(images, list):
        images = [images]

    if prompt_style is not None:
        content = build_guardian_execution_content(
            images=images,
            task=task,
            prompt_style=prompt_style,
        )
    else:
        content = [{"type": "image", "image": image} for image in images]
        content.append(
            {
                "type": "text",
                "text": (
                    "You are a robot task failure detector. "
                    f"Evaluate whether the robot successfully completed the task: {task}. "
                    "Answer only success or fail."
                ),
            }
        )

    return [{"role": "user", "content": content}]


def validate_model(model, val_dataset, batch_size, prediction_mode="fusion"):
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i in range(0, len(val_dataset), batch_size):
            batch_end = min(i + batch_size, len(val_dataset))
            entries = val_dataset[i:batch_end]
            try:
                batch_images = entries["images"]
                batch_tasks = entries["task"]
                batch_labels = [label_to_binary(label) for label in entries["label"]]

                if "prompt_style" in entries:
                    batch_prompt_styles = entries["prompt_style"]
                else:
                    batch_prompt_styles = [None] * len(batch_tasks)

                predictions, _ = model.predict(
                    batch_images,
                    batch_tasks,
                    prompt_styles=batch_prompt_styles,
                    voting=False,
                    prediction_mode=prediction_mode,
                )
                batch_labels = torch.tensor(
                    batch_labels, dtype=torch.float32, device=model.device
                )
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)
            except Exception as exc:
                print(f"[Warning] Skipping validation batch due to error: {exc}")
                continue

    return val_correct / val_total if val_total > 0 else 0.0


class TaskConditionedPooling(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.base_query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.task_to_query = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.out_norm = nn.LayerNorm(input_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, text_mask=None, attention_mask=None):
        batch_size, _, _ = x.shape

        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        mask_expanded = attention_mask.unsqueeze(-1).bool() if attention_mask is not None else None

        text_mask_expanded = text_mask.unsqueeze(-1).to(x.dtype)
        task_repr = (x * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1).clamp(min=1)

        query_shift = self.task_to_query(task_repr).unsqueeze(1)
        query = self.base_query.expand(batch_size, -1, -1) + query_shift

        mlp_scores = self.attn_mlp(x)
        _, mha_weights = self.mha(
            query,
            x,
            x,
            key_padding_mask=key_padding_mask,
        )
        mha_scores = mha_weights.transpose(1, 2)

        combined_scores = self.alpha * mlp_scores + (1 - self.alpha) * mha_scores
        combined_scores -= combined_scores.max(dim=1, keepdim=True).values

        if mask_expanded is not None:
            combined_scores = combined_scores.masked_fill(
                ~mask_expanded,
                torch.finfo(x.dtype).min,
            )

        weights = F.softmax(combined_scores, dim=1)
        weights = self.attn_dropout(weights)
        pooled = torch.sum(weights * x, dim=1)
        pooled = self.out_norm(pooled)

        return pooled, task_repr, weights.squeeze(-1)


class HybridAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, _, _ = x.shape

        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        mask_expanded = attention_mask.unsqueeze(-1).bool() if attention_mask is not None else None

        mlp_scores = self.attn_mlp(x)
        query = self.query.expand(batch_size, -1, -1)
        _, mha_weights = self.mha(query, x, x, key_padding_mask=key_padding_mask)
        mha_scores = mha_weights.transpose(1, 2)

        combined_scores = self.alpha * mlp_scores + (1 - self.alpha) * mha_scores
        combined_scores -= combined_scores.max(dim=1, keepdim=True).values

        if mask_expanded is not None:
            combined_scores = combined_scores.masked_fill(
                ~mask_expanded,
                torch.finfo(x.dtype).min,
            )

        weights = F.softmax(combined_scores, dim=1)
        weights = self.attn_dropout(weights)
        pooled = torch.sum(weights * x, dim=1)

        return pooled, weights.squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.act(x)
        x = self.fc(x)
        x = self.dropout(x)
        return residual + x


class MLP_BLOCK(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate=0.4):
        super().__init__()
        self.residual = ResidualBlock(dim_in, dropout_rate)
        self.norm = nn.LayerNorm(dim_in)
        self.act = nn.GELU()
        self.fc = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.residual(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class LearnedLayerFusion(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_features):
        weights = torch.softmax(self.layer_logits, dim=0)
        fused = torch.sum(layer_features * weights.view(1, -1, 1), dim=1)
        return fused, weights


class DynamicLayerFusion(nn.Module):
    def __init__(self, num_layers, dim, hidden_dim=256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, layer_features):
        scores = self.scorer(layer_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(layer_features * weights.unsqueeze(-1), dim=1)
        return fused, weights.mean(dim=0)


class QwenFailSenseMultiLayerFusion(nn.Module):
    def __init__(
        self,
        vlm_model_id,
        device="cuda",
        dropout_rate=0.1,
        num_classifiers=3,
        target_layer_indices=None,
        pooling_mode="tcond",
        fusion_mode="static",
    ):
        super().__init__()

        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device(device)
        self.num_classifiers = num_classifiers
        self.dropout_rate = dropout_rate
        self.pooling_mode = pooling_mode
        self.fusion_mode = fusion_mode

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        print(f"Loading Qwen LoRA adapter from checkpoint {vlm_model_id}")

        self.vlm_model = PeftModel.from_pretrained(base_model, vlm_model_id)
        self.vlm_model.to(self.device)
        self.vlm_model.eval()

        for param in self.vlm_model.parameters():
            param.requires_grad = False

        config = self.vlm_model.base_model.model.config
        if hasattr(config, "text_config"):
            feature_dim = config.text_config.hidden_size
        else:
            feature_dim = config.hidden_size

        print(f"Qwen hidden size: {feature_dim}")

        if target_layer_indices is None:
            target_layer_indices = [13, 25, 36]
        if len(target_layer_indices) != self.num_classifiers:
            raise ValueError(
                f"target_layer_indices has {len(target_layer_indices)} entries, "
                f"but num_classifiers={self.num_classifiers}"
            )
        self.target_layer_indices = target_layer_indices
        print(f"Using Qwen hidden states: {self.target_layer_indices}")
        print(f"Pooling mode: {self.pooling_mode}")

        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)

        self.att_poolings = nn.ModuleList(
            [
                TaskConditionedPooling(input_dim=feature_dim, dropout=dropout_rate).to(
                    self.device
                )
                for _ in range(self.num_classifiers)
            ]
        )
        self.hybrid_poolings = nn.ModuleList(
            [
                HybridAttentionPooling(input_dim=feature_dim, dropout=dropout_rate).to(
                    self.device
                )
                for _ in range(self.num_classifiers)
            ]
        )
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    MLP_BLOCK(feature_dim, 1024, dropout_rate),
                    MLP_BLOCK(1024, 256, dropout_rate),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, 1),
                ).to(self.device)
                for _ in range(self.num_classifiers)
            ]
        )

        if fusion_mode == "dynamic":
            self.layer_fusion = DynamicLayerFusion(self.num_classifiers, feature_dim).to(
                self.device
            )
        else:
            self.layer_fusion = LearnedLayerFusion(self.num_classifiers).to(self.device)

        self.fused_classifier = nn.Sequential(
            MLP_BLOCK(feature_dim, 1024, dropout_rate),
            MLP_BLOCK(1024, 256, dropout_rate),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        ).to(self.device)

    def pool_features(self, features, layer_idx, text_mask=None, attention_mask=None):
        if self.pooling_mode == "tcond":
            pooled, _, _ = self.att_poolings[layer_idx](
                features,
                text_mask=text_mask,
                attention_mask=attention_mask,
            )
            return pooled
        if self.pooling_mode == "hybrid":
            pooled, _ = self.hybrid_poolings[layer_idx](
                features,
                attention_mask=attention_mask,
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

    def build_qwen_prompts(self, images_batch, tasks, prompt_styles=None):
        if prompt_styles is None:
            prompt_styles = [None] * len(tasks)

        prompts = []
        flat_images = []

        for images, task, prompt_style in zip(images_batch, tasks, prompt_styles):
            if not isinstance(images, list):
                images = [images]

            messages = build_messages(images=images, task=task, prompt_style=prompt_style)
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            flat_images.extend(images)

        return prompts, flat_images

    def make_text_mask(self, model_inputs):
        input_ids = model_inputs["input_ids"]
        text_mask = model_inputs["attention_mask"].bool()
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        if image_token_id is not None:
            text_mask = text_mask & (input_ids != image_token_id)
        if video_token_id is not None:
            text_mask = text_mask & (input_ids != video_token_id)
        return text_mask

    def extract_features(self, images, tasks, prompt_styles=None, voting=False):
        prompts, flat_images = self.build_qwen_prompts(images, tasks, prompt_styles)

        try:
            model_inputs = self.processor(
                text=prompts,
                images=flat_images,
                return_tensors="pt",
                padding=True,
            )

            for key, value in model_inputs.items():
                if not torch.is_tensor(value):
                    continue
                if key in ["image_grid_thw", "video_grid_thw"]:
                    model_inputs[key] = value.cpu()
                elif key == "pixel_values":
                    model_inputs[key] = value.to(self.device, dtype=torch.bfloat16)
                else:
                    model_inputs[key] = value.to(self.device)
        except Exception as exc:
            raise RuntimeError(f"Error processing Qwen inputs: {exc}") from exc

        with torch.no_grad():
            vlm_output = self.vlm_model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        decoded = None
        if voting:
            logits = vlm_output.logits
            attention_mask = model_inputs["attention_mask"].to(logits.device)
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            last_token_logits = logits[batch_indices, last_token_indices, :]
            predicted_token_id = torch.argmax(last_token_logits, dim=-1)
            decoded = [
                self.processor.decode([token_id.item()], skip_special_tokens=True)
                for token_id in predicted_token_id
            ]

        hidden_states = vlm_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states.")

        features = []
        for idx in self.target_layer_indices:
            if idx >= len(hidden_states):
                raise RuntimeError(
                    f"Requested hidden state index {idx}, "
                    f"but Qwen returned only {len(hidden_states)} hidden states."
                )
            features.append(
                hidden_states[idx].detach().to(device=self.device, dtype=torch.float32)
            )

        attention_mask = model_inputs["attention_mask"].to(self.device)
        text_mask = self.make_text_mask(model_inputs).to(self.device)

        return decoded, features, text_mask, attention_mask

    def forward(self, images, tasks, prompt_styles=None, voting=False):
        decoded, all_layer_features, text_mask, attention_mask = self.extract_features(
            images=images,
            tasks=tasks,
            prompt_styles=prompt_styles,
            voting=voting,
        )

        classifier_outputs = []
        pooled_layer_features = []

        for i in range(self.num_classifiers):
            features = all_layer_features[i]
            if len(features.shape) != 3:
                raise ValueError(f"Expected 3D features [B, T, D], got shape {features.shape}")

            pooled = self.pool_features(
                features,
                layer_idx=i,
                text_mask=text_mask,
                attention_mask=attention_mask,
            )
            logit = self.classifiers[i](pooled)
            classifier_outputs.append(logit)
            pooled_layer_features.append(pooled)

        layer_features = torch.stack(pooled_layer_features, dim=1)
        fused, _ = self.layer_fusion(layer_features)
        fused_logit = self.fused_classifier(fused)
        classifier_outputs.append(fused_logit)

        if voting:
            return decoded, classifier_outputs
        return classifier_outputs

    def predict(
        self,
        images,
        tasks,
        prompt_styles=None,
        voting=False,
        prediction_mode="fusion",
    ):
        self.eval()
        with torch.no_grad():
            if prediction_mode == "fusion":
                logits = self.forward(
                    images,
                    tasks,
                    prompt_styles=prompt_styles,
                    voting=False,
                )
                fusion_logit = logits[-1]
                fusion_prob = torch.sigmoid(fusion_logit.squeeze(-1))
                predictions = (fusion_prob > 0.5).float()
                return predictions, fusion_prob

            logits = self.forward(
                images,
                tasks,
                prompt_styles=prompt_styles,
                voting=voting,
            )
            if voting:
                decoded, logits = logits

            head_logits = logits[: self.num_classifiers]
            head_probs = torch.stack(
                [torch.sigmoid(logit.squeeze(-1)) for logit in head_logits],
                dim=0,
            )

            if prediction_mode == "head_average":
                avg_probs = head_probs.mean(dim=0)
                return (avg_probs > 0.5).float(), avg_probs

            if prediction_mode == "head_majority":
                votes = (head_probs > 0.5).float().sum(dim=0)
                threshold = (self.num_classifiers // 2) + 1
                avg_probs = votes / self.num_classifiers
                return (votes >= threshold).float(), avg_probs

            if prediction_mode == "head_plus_vlm_vote":
                if not voting:
                    _, logits_with_vote = self.forward(
                        images,
                        tasks,
                        prompt_styles=prompt_styles,
                        voting=True,
                    )
                    decoded, logits = logits_with_vote
                head_probs = torch.stack(
                    [torch.sigmoid(logit.squeeze(-1)) for logit in logits[: self.num_classifiers]],
                    dim=0,
                )
                vote_tensor = (head_probs > 0.5).float()
                device = vote_tensor.device
                predictions = []
                avg_probs = []
                for i, dec in enumerate(decoded):
                    clf_score = vote_tensor[:, i].sum()
                    d = dec.strip().lower()
                    if d in ["success", "1", "pass"]:
                        vlm_vote = 1
                    elif d in ["fail", "0", "failure"]:
                        vlm_vote = 0
                    else:
                        vlm_vote = None

                    if vlm_vote is None:
                        total_score = clf_score
                        threshold = 2
                        max_score = 3
                    else:
                        total_score = clf_score + (2 * vlm_vote)
                        threshold = 3
                        max_score = 5

                    predictions.append(1.0 if total_score >= threshold else 0.0)
                    avg_probs.append(total_score / max_score)

                return (
                    torch.tensor(predictions, device=device),
                    torch.tensor(avg_probs, device=device),
                )

            raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    def cleanup(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def save_classifier(self, path="./checkpoints", epoch=None):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "num_classifiers": self.num_classifiers,
            "dropout_rate": self.dropout_rate,
            "target_layer_indices": self.target_layer_indices,
            "pooling_mode": self.pooling_mode,
            "fusion_mode": self.fusion_mode,
            "layer_fusion": self.layer_fusion.state_dict(),
            "fused_classifier": self.fused_classifier.state_dict(),
        }
        for i in range(self.num_classifiers):
            checkpoint[f"classifier_{i}"] = self.classifiers[i].state_dict()
            checkpoint[f"attention_pooling_{i}"] = self.att_poolings[i].state_dict()
            checkpoint[f"hybrid_pooling_{i}"] = self.hybrid_poolings[i].state_dict()

        filename = "components.pt" if epoch is None else f"components_epoch_{epoch}.pt"
        if epoch is not None:
            checkpoint["epoch"] = epoch
        full_path = os.path.join(path, filename)
        torch.save(checkpoint, full_path)
        print(f"Saved Qwen fusion components to {full_path}")

    def load_classifier(self, path, strict=True):
        assert os.path.isfile(path), f"Checkpoint not found: {path}"
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        for i in range(self.num_classifiers):
            self.classifiers[i].load_state_dict(
                checkpoint[f"classifier_{i}"], strict=strict
            )
            self.att_poolings[i].load_state_dict(
                checkpoint[f"attention_pooling_{i}"], strict=strict
            )
            if f"hybrid_pooling_{i}" in checkpoint:
                self.hybrid_poolings[i].load_state_dict(
                    checkpoint[f"hybrid_pooling_{i}"], strict=strict
                )

        self.layer_fusion.load_state_dict(checkpoint["layer_fusion"], strict=strict)
        self.fused_classifier.load_state_dict(checkpoint["fused_classifier"], strict=strict)
        return checkpoint.get("epoch")


def train_model(model, train_dataset, val_dataset, config):
    criterion = nn.BCEWithLogitsLoss()

    trainable_params = []
    for i in range(model.num_classifiers):
        trainable_params.extend(model.classifiers[i].parameters())
        if model.pooling_mode == "tcond":
            trainable_params.extend(model.att_poolings[i].parameters())
        if model.pooling_mode == "hybrid":
            trainable_params.extend(model.hybrid_poolings[i].parameters())
    trainable_params.extend(model.layer_fusion.parameters())
    trainable_params.extend(model.fused_classifier.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    steps_per_epoch = max(1, len(train_dataset) // config["batch_size"])
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
        model.vlm_model.eval()

        running_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0

        progress_bar = tqdm(
            range(0, len(train_dataset), config["batch_size"]),
            desc=f"Epoch {epoch + 1}",
        )

        for start in progress_bar:
            end = min(start + config["batch_size"], len(train_dataset))
            if end - start < config["batch_size"]:
                continue

            entries = train_dataset[start:end]
            tasks = entries["task"]
            images = entries["images"]
            labels = torch.tensor(
                [label_to_binary(label) for label in entries["label"]],
                dtype=torch.float32,
                device=model.device,
            )
            prompt_styles = entries.get("prompt_style", [None] * len(tasks))

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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1

            with torch.no_grad():
                prob = torch.sigmoid(logits[-1].squeeze(-1))
                predictions = (prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct / total:.4f}" if total else "0.0000",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                }
            )

        val_acc = validate_model(
            model,
            val_dataset,
            config["batch_size"],
            prediction_mode=prediction_mode,
        )
        train_acc = correct / total if total else 0.0
        avg_loss = running_loss / num_batches if num_batches else 0.0
        print(
            f"  end of epoch {epoch + 1}: train_loss={avg_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_classifier(
                path=config["save_path"],
                epoch=f"best_epoch_{epoch + 1}",
            )
            print(f"  new best val_acc = {best_val_acc:.4f}")

        model.save_classifier(path=config["save_path"], epoch=epoch + 1)

    print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
    return best_val_acc
