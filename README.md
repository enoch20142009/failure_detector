# I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models

I-FailSense is a **vision-language model (VLM)** for detecting **language-conditioned robotic failures** from visual observations. This repository contains code for training and evaluating I-FailSense models on robotic datasets DROID, Calvin and AHA.

📄 **Paper:** [https://arxiv.org/abs/2509.16072](https://arxiv.org/abs/2509.16072)

**Conference:** ICRA 2026 🎉

---

## 🚀 Installation

Create a Python environment and install dependencies:

```bash
conda create -n ifailsense python=3.10
conda activate ifailsense
pip install -r requirements.txt
```

---

## 🧪 Evaluation

Pre-trained weights are available for evaluation:

* **LoRA weights:** available on [Hugging Face](https://huggingface.co/collections/ACIDE/failsense-3b)
* **FS block weights:** can be downloaded via `wget`

```bash
wget -c https://github.com/clemgris/I-FailSense/releases/tag/models/FS_blocks.zip
```

Run evaluation with:

```bash
python src/evaluate.py \
    --vlm_model_id ACIDE/FailSense-Calvin-1p-3b \
    --fs_id FS/FailSense-Calvin-1p-3b \
    --dataset_name calvin \
    --result_folder results_calvin_1p
```

This will evaluate the model on the **Calvin dataset** and save results in the specified folder.

---

## 🏋️ Training

### Phase 1: Fine-tuning the base VLM with LoRA

```bash
python src/finetune_VLM.py \
    --pov 1 \
    --batch_size 4 \
    --num_epochs 3
```

### Phase 2: Training the FS Blocks

```bash
python src/finetune_FS.py \
    --dataset_name droid \
    --vlm_model_id ACIDE/FailSense-Calvin-1p-3b \
    --batch_size 4 \
    --num_epochs 10
```

---

## OneVision-2 Routed Extension (MaTCA = eval pipeline name only)

This repository also contains an experimental extension on a frozen **LLaVA-OneVision-2-8B-Instruct**
backbone with optional **pre-LLM Stage-1** task/failure routing and multilayer fusion (NGF, MoE).

**MaTCA** here names a **personal post-LLM eval pipeline** (task-conditioned pooling +
fusion MLP, I-FailSense-inspired, fusion instead of voting) — **not** a claimed novel method.
Paper contributions target **Stage-1 pre-LLM grounding/fusion** only.

**Documentation (read in this order):**

| Doc | Contents |
| --- | --- |
| [`NGF_ARCH_C_SUPERVISOR_QA.md`](NGF_ARCH_C_SUPERVISOR_QA.md) | **Supervisor Q&A** — Arch C architecture, code map, training loop, param counts |
| [`NGF_0_V2.md`](NGF_0_V2.md) | **NGF-0 v2 only** — parallel nested fusion formulas, inner/outer loops, run config & results |
| [`NGF_ARCH_C.md`](NGF_ARCH_C.md) | **Arch C only** — sequential NGF formulas, architecture, your run config & results |
| [`OV2_ROUTED_ARCHITECTURE.md`](OV2_ROUTED_ARCHITECTURE.md) | Module layout, data flow, NGF v2 / Arch B / Arch C, ablation ladder, results |
| [`OV2_ROUTING_TRAINING.md`](OV2_ROUTING_TRAINING.md) | GADI setup, CLI flags, job matrix, go/no-go criteria |
| [`EXPERIMENT_RUN_SCHEDULE.md`](EXPERIMENT_RUN_SCHEDULE.md) | Full run registry (Tracks A/B/C), flag cheat sheet |
| [`SUMMARY_OV2_RESULTS.md`](SUMMARY_OV2_RESULTS.md) | Baseline/routed/MoE matrix + NGF results |
| [`PAPER_PLAN.md`](PAPER_PLAN.md) | Paper narrative, method IDs M0–M5, experiment phases |
| [`PUBLISH_READINESS_CHECKLIST.md`](PUBLISH_READINESS_CHECKLIST.md) | Pre-submission checklist |

Key entry points:

| Script | Purpose |
| --- | --- |
| `src/model_ov2_routed_matca.py` | Model (Stage-1 routing/NGF/MoE + fixed post-LLM eval pipeline) |
| `src/finetune_FS_ov2_routed.py` | Training driver |
| `src/evaluate_FS_ov2_routed.py` | Evaluation + grounding probe |
| `gadi_scripts/ov2_routing/` | PBS jobs for GADI (`gpuhopper` + `dgxa100`) |

---

## 🔗 References

```
@inproceedings{ifailsense2026,
  title        = {I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models},
  author       = {Clemence Grislain and Hamed Rahimi and Olivier Sigaud and Mohamed Chetouani},
  booktitle    = {Proceedings of the International Conference on Robotics and Automation (ICRA)},
  year         = {2026},
  url          = {https://arxiv.org/abs/2509.16072}
}
```
