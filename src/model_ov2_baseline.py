"""LLaVA-OneVision-2 baseline for binary robot outcome prediction.

This module contains the model-facing half of the OV2 evaluation pipeline. It
normalizes labels, builds image/text chat prompts, loads the pinned checkpoint,
prepares multimodal batches, and produces ``fail`` (0) or ``success`` (1)
predictions using either text generation or first-token logits.

The dataset iteration, metrics, and result serialization are implemented in
``evaluate_VLM_ov2.py``.
"""

import re
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_OV2_MODEL_ID = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
DEFAULT_OV2_REVISION = "5a75eaf7d3cd73de6f85e637e45b420f46857d2e"


def label_to_binary(label):
    """Map dataset labels to the shared convention ``fail=0, success=1``."""
    return 0 if label in ("0", "fail", 0, False) else 1


def build_guardian_execution_content(images, task, prompt_style):
    """Build temporally and spatially labeled Guardian prompt content.

    Images must be ordered as all start-state views followed by all end-state
    views. Every ``image`` placeholder is paired with the next image supplied
    to the processor.
    """
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
                {"type": "text", "text": "\nImage at the end of the subtask:"},
                {"type": "image", "image": images[1]},
            ]
        )

    elif prompt_style == "three_viewpoints":
        if len(images) != 6:
            raise ValueError(f"three_viewpoints expects 6 images, got {len(images)}")

        view_names = ["Front view", "Left view", "Right view"]

        content.append(
            {"type": "text", "text": "Multiview images at the start of the subtask:"}
        )
        for view_name, image in zip(view_names, images[:3]):
            content.extend(
                [
                    {"type": "text", "text": f"\n{view_name}:"},
                    {"type": "image", "image": image},
                ]
            )

        content.append(
            {"type": "text", "text": "\nMultiview images at the end of the subtask:"}
        )
        for view_name, image in zip(view_names, images[3:]):
            content.extend(
                [
                    {"type": "text", "text": f"\n{view_name}:"},
                    {"type": "image", "image": image},
                ]
            )

    elif prompt_style == "four_viewpoints":
        if len(images) != 8:
            raise ValueError(f"four_viewpoints expects 8 images, got {len(images)}")

        view_names = ["Left view", "Right view", "Wrist view", "Front view"]

        content.append(
            {"type": "text", "text": "Multiview images at the start of the subtask:"}
        )
        for view_name, image in zip(view_names, images[:4]):
            content.extend(
                [
                    {"type": "text", "text": f"\n{view_name}:"},
                    {"type": "image", "image": image},
                ]
            )

        content.append(
            {"type": "text", "text": "\nMultiview images at the end of the subtask:"}
        )
        for view_name, image in zip(view_names, images[4:]):
            content.extend(
                [
                    {"type": "text", "text": f"\n{view_name}:"},
                    {"type": "image", "image": image},
                ]
            )

    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    content.append(
        {
            "type": "text",
            "text": (
                f"\nDetermine whether the robot successfully completed this subtask: {task}"
                "\nAnswer with exactly one word: success or fail."
            ),
        }
    )

    return content


def build_messages(images, task, prompt_style=None):
    """Create one OV2-compatible user message for an evaluation sample."""
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
                    f"Determine whether the robot successfully completed this task: {task}"
                    "\nAnswer with exactly one word: success or fail."
                ),
            }
        )

    return [{"role": "user", "content": content}]


def parse_binary_response(response):
    """Return 1 for success, 0 for failure, or ``None`` if no class is found."""
    normalized = response.strip().lower()
    match = re.search(r"\b(success|successful|fail|failed|failure)\b", normalized)
    if not match:
        return None
    token = match.group(1)
    if token in ("success", "successful"):
        return 1
    return 0


class OV2Baseline:
    """Load OV2 and expose batched binary inference methods."""

    def __init__(
        self,
        model_id=DEFAULT_OV2_MODEL_ID,
        revision=DEFAULT_OV2_REVISION,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=None,
        max_pixels=200704,
    ):
        """Initialize the processor and inference-only model.

        ``device_map`` controls Accelerate placement, ``dtype`` controls model
        weight precision, and ``max_pixels`` limits visual preprocessing cost.
        """
        self.model_id = model_id
        self.revision = revision

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )

        if max_pixels is not None:
            self.processor.image_processor.max_pixels = int(max_pixels)
            if hasattr(self.processor.image_processor, "size"):
                self.processor.image_processor.size["longest_edge"] = int(max_pixels)

        model_kwargs = {
            "revision": revision,
            "trust_remote_code": True,
            "dtype": dtype,
            "device_map": device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **model_kwargs,
        ).eval()

    @property
    def input_device(self):
        """Return the device that should receive processor-produced tensors."""
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def prepare_batch(self, images_batch, tasks, prompt_styles=None):
        """Render prompts, flatten images, tokenize, and place a batch on-device."""
        if prompt_styles is None:
            prompt_styles = [None] * len(tasks)

        prompts = []
        flat_images = []

        for images, task, prompt_style in zip(images_batch, tasks, prompt_styles):
            if not isinstance(images, list):
                images = [images]

            messages = build_messages(
                images=images,
                task=task,
                prompt_style=prompt_style,
            )
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

        return {
            key: value.to(self.input_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def _predict_generate(self, inputs, max_new_tokens=8):
        """Generate short answers and parse them into binary predictions."""
        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        responses = [
            tokenizer.decode(
                output_ids[i, prompt_length:],
                skip_special_tokens=True,
            )
            for i in range(output_ids.shape[0])
        ]

        predictions = []
        probabilities = []
        for response in responses:
            parsed = parse_binary_response(response)
            if parsed is None:
                predictions.append(0.0)
                probabilities.append(0.0)
            else:
                predictions.append(float(parsed))
                probabilities.append(float(parsed))

        return (
            torch.tensor(predictions, device=self.input_device),
            torch.tensor(probabilities, device=self.input_device),
            responses,
        )

    def _predict_first_token(self, inputs):
        """Classify from next-token logits for the words ``fail`` and ``success``."""
        tokenizer = self.processor.tokenizer
        fail_ids = tokenizer.encode("fail", add_special_tokens=False)
        success_ids = tokenizer.encode("success", add_special_tokens=False)

        if len(fail_ids) != 1 or len(success_ids) != 1:
            raise ValueError(
                "first_token mode requires single-token labels. "
                f"fail_ids={fail_ids}, success_ids={success_ids}. "
                "Use decision_mode='generate' instead."
            )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        attention_mask = inputs["attention_mask"]
        last_indices = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
        next_token_logits = outputs.logits[batch_indices, last_indices, :]

        fail_id = fail_ids[0]
        success_id = success_ids[0]
        binary_logits = torch.stack(
            [
                next_token_logits[:, fail_id],
                next_token_logits[:, success_id],
            ],
            dim=1,
        )
        binary_probs = torch.softmax(binary_logits, dim=1)
        predictions = (binary_probs[:, 1] > binary_probs[:, 0]).float()
        probabilities = binary_probs[:, 1]
        responses = ["success" if p.item() > 0.5 else "fail" for p in predictions]

        return predictions, probabilities, responses

    def predict(
        self,
        images_batch,
        tasks,
        prompt_styles=None,
        decision_mode="generate",
        max_new_tokens=8,
    ):
        """Prepare a batch and dispatch to the selected decision strategy."""
        inputs = self.prepare_batch(images_batch, tasks, prompt_styles=prompt_styles)

        if decision_mode == "generate":
            predictions, probabilities, responses = self._predict_generate(
                inputs,
                max_new_tokens=max_new_tokens,
            )
            return predictions, probabilities, responses

        if decision_mode == "first_token":
            predictions, probabilities, responses = self._predict_first_token(inputs)
            return predictions, probabilities, responses

        raise ValueError(f"Unknown decision_mode: {decision_mode}")
