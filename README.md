<h1 align="center">
  <img src="assets/logo.png" width="60" style="vertical-align: middle; margin-right: 8px;">
  OrionEdit: Bridging Reference and Source Images for Generalized Cross-Image Editing
</h1>

<p align="center">
    <a href="https://github.com/cityuhkai/OrionEdit">
        <img alt="Paper" src="https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey">
    </a>
    <a href="https://cityuhkai.github.io/OrionEdit/">
        <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue">
    </a>
    <a href="https://github.com/cityuhkai/OrionEdit">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black">
    </a>
    <a href="https://huggingface.co/ZeyuJiang1/OrionEdit-qwen">
        <img alt="Build" src="https://img.shields.io/badge/🤗-HF%20Model-yellow">
    </a>
    <a href="https://huggingface.co/datasets/ZeyuJiang1/OrionEditBench">
        <img alt="Build" src="https://img.shields.io/badge/🤗-HF%20Dataset-yellow">
    </a>    
</p>

![The teaser figure of OrionEdit.](assets/teaser.png)


## 🔥 News

- **2026.3.22**: The repo has been released!
- **2026.4.01**: We release a subset of the OrionEditBench metadata, including the AI-generated data for attribute transfer task!
- **2026.4.06**: We release the inference code of OrionEdit-qwen model!
- **2026.4.19**: We release the training example of OrionEdit-qwen model!


## 📖 Introduction

We present OrionEdit, a unified framework for cross-image editing that combines symmetric orthogonal subspace disentanglement with reverse-causal attention, where information-flow masks enforce unidirectional dependencies in the latent space.

OrionEdit is deployed on standard diffusion backbones and supports zero-shot multi-reference editing, while outperforming open-source baselines in fidelity and compositional consistency.




## 🚀 Quick Start

### Environment

We recommend a CUDA GPU with **≥24 GB** VRAM for inference (the pipeline enables sequential CPU offload when a device is set). Install dependencies via Conda:

```bash
conda env create -f config/exvironment.yml
conda activate orionedit
```

Key versions: Python 3.10, PyTorch 2.9, `diffusers==0.36.0.dev0`, `transformers==4.57.1`, `peft==0.17.1`. Log in to Hugging Face if needed (`huggingface-cli login`) so weights for [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) and [ZeyuJiang1/OrionEdit-qwen](https://huggingface.co/ZeyuJiang1/OrionEdit-qwen) can be downloaded.

### Inference

The demo script [`inference.py`](inference.py) loads the base Qwen backbone plus OrionEdit LoRA and decoupling processors via `OrionEditPipeline.from_orion_pretrained`:

```bash
python inference.py
```

Edit the paths and prompt at the top of `inference.py` before running. Minimal usage in code:

```python
from PIL import Image
import torch
from models.pipeline_orion_edit import OrionEditPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

pipeline = OrionEditPipeline.from_orion_pretrained(
    base_model="Qwen/Qwen-Image-Edit-2511",
    orion_repo="ZeyuJiang1/OrionEdit-qwen",
    torch_dtype=dtype,
    device=device,
)
pipeline = pipeline.to(dtype=torch.bfloat16)

# --- Attribute transfer / editing (1–3 references + one source scene) ---
reference_image = Image.open("example1-ref.png").convert("RGB")
source_image = Image.open("example1-source.png").convert("RGB")

image = pipeline(
    prompt = "replace the character in Figure 2 with the character in Figure 1.",
    reference_image=reference_image,
    source_image=source_image,
    num_inference_steps=30,
    true_cfg_scale=4.0,
    negative_prompt=" ",
    guidance_scale=1.0,
).images[0]

image.save("example1-output.png")
```

**Multi-reference input (2–3 references):**

```python
from PIL import Image
import torch
from models.pipeline_orion_edit import OrionEditPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

pipeline = OrionEditPipeline.from_orion_pretrained(
    base_model="Qwen/Qwen-Image-Edit-2511",
    orion_repo="ZeyuJiang1/OrionEdit-qwen",
    torch_dtype=dtype,
    device=device,
)
pipeline = pipeline.to(dtype=torch.bfloat16)

# --- Attribute transfer / editing (1–3 references + one source scene) ---
reference_image = [Image.open("example3-ref1.png").convert("RGB"), Image.open("example3-ref2.png").convert("RGB")]
source_image = Image.open("example3-source.png").convert("RGB")

image = pipeline(
    prompt = "characters in Figure 1 walking on the sunset street in Figure 2, with their backs facing the camera, anime style.",
    reference_image=reference_image,
    source_image=source_image,
    num_inference_steps=30,
    true_cfg_scale=4.0,
    negative_prompt=" ",
    guidance_scale=1.0,
).images[0]

image.save("example3-output.png")
```

**Fusion (two references, no source):**

```python
from PIL import Image
import torch
from models.pipeline_orion_edit import OrionEditPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

pipeline = OrionEditPipeline.from_orion_pretrained(
    base_model="Qwen/Qwen-Image-Edit-2511",
    orion_repo="ZeyuJiang1/OrionEdit-qwen",
    torch_dtype=dtype,
    device=device,
)
pipeline = pipeline.to(dtype=torch.bfloat16)

# --- Fusion (two references, no source) ---
reference_image = [Image.open("example4-ref1.png").convert("RGB"), Image.open("example4-ref2.png").convert("RGB")]
source_image = None

image = pipeline(
    prompt = "photo of two men walking on the street, they talking with eacher other.",
    reference_image=reference_image,
    source_image=source_image,
    num_inference_steps=30,
    width=1024,
    height=1024,
    true_cfg_scale=4.0,
    negative_prompt=" ",
    guidance_scale=1.0,
).images[0]

image.save("example4-output.png")
```

| Task | `reference_image` | `source_image` | Notes |
|------|-------------------|----------------|-------|
| **Editing** (attribute transfer, style alignment, etc.) | 1–3 images | One scene to edit | Multiple refs are composited left-to-right against the source before encoding. |
| **Fusion** | Exactly 2 images | `None` or `""` | Generates a new image from two references without a source scene. |

Default inference knobs in `inference.py`: `num_inference_steps=30`, `true_cfg_scale=4.0`, `guidance_scale=1.0`.

### Training

Training extends the [diffusers Qwen-Image-Edit](https://github.com/huggingface/diffusers) loop with OrionEdit-specific modules (`OrionEditTransformer2DModel`, decoupling processors, and `CustomDataset`). Defaults live in [`config/base.py`](config/base.py); override them on the CLI.

1. Prepare data under `dataset/` (see [`dataset/readme.txt`](dataset/readme.txt)): `metadata/metadata.json` plus images referenced by relative paths.
2. Point `--train-metadata-json` to your metadata file and `--pretrained-model-name-or-path` to `Qwen/Qwen-Image-Edit-2511` (or a local clone).
3. Launch training:

```bash
python train.py \
  --pretrained-model-name-or-path Qwen/Qwen-Image-Edit-2511 \
  --train-metadata-json dataset/metadata/metadata.json \
  --output-dir ./output \
  --max-train-steps 3000 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 6 \
  --learning-rate 7e-5 \
  --rank 256 \
  --mixed-precision bf16
```

Exactly **one** data source must be set: `--train-metadata-json`, `--dataset-name`, or `--jsonl-for-train`. Checkpoints write LoRA and processor weights compatible with `from_orion_pretrained`. For dataset field definitions and edit vs. fusion rules during training, see the [dataset readme](dataset/readme.txt) and `CustomDataset` in [`models/custom_dataset.py`](models/custom_dataset.py).


## 🤗 Models

### Released checkpoints

| Model | Hugging Face | Base backbone | Description |
|-------|--------------|---------------|-------------|
| **OrionEdit-qwen** | [ZeyuJiang1/OrionEdit-qwen](https://huggingface.co/ZeyuJiang1/OrionEdit-qwen) | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Our main released weights for cross-image editing and fusion on the Qwen-Image-Edit family. |

`from_orion_pretrained` downloads two artifacts from **OrionEdit-qwen**:

- `pytorch_lora_weights.safetensors` — LoRA adapters on the diffusion transformer (rank 256).
- `pytorch_processor_weights.safetensors` — Symmetric orthogonal subspace decoupling processors attached to attention blocks (`d=3072`, rank 256).

The VAE, text encoder (Qwen2.5-VL), tokenizer, and scheduler are loaded from **Qwen-Image-Edit-2511**; only the transformer is specialized via `OrionEditTransformer2DModel` and the weights above.

> **Note:** Experiments in the paper used Qwen-Image-Edit-**2509**; this repository trains and infers on **2511** for compatibility with the latest diffusers pipeline.

### Code map

| Component | Path | Role |
|-----------|------|------|
| `OrionEditPipeline` | [`models/pipeline_orion_edit.py`](models/pipeline_orion_edit.py) | Multi-reference editing & fusion inference; `from_orion_pretrained` loader. |
| `OrionEditTransformer2DModel` | [`models/transformer_orion.py`](models/transformer_orion.py) | Transformer with DNS decoupling processors and reverse-causal attention masks. |
| `CustomDataset` | [`models/custom_dataset.py`](models/custom_dataset.py) | JSON metadata loader for training triplets. |
| Training entry | [`train.py`](train.py) | Fine-tuning with LoRA + decoupling trainables. |
| Inference demo | [`inference.py`](inference.py) | Minimal end-to-end example. |

### Loading a custom checkpoint

After training, point `orion_repo` to a local folder or Hub repo that contains the same two safetensors filenames, or call `load_lora_weights` / load processor state dict manually following [`from_orion_pretrained`](models/pipeline_orion_edit.py).


## 🗂️ OrionEditBench

We construct a dataset based on **reference–source–synthesis triplets** to support cross-image editing. Due to the lack of large-scale data in this format, our collection combines samples adapted from existing public datasets (e.g., Subjects200K, ShareGPT-4o-Image, OmniContext, DeepFashion) with a substantial portion of curated synthetic pairs generated using Nano-banana and GPT-4o.

The released subset covers diverse editing scenarios, with a focus on **visual attribute transfer**, along with fusion-based generation and style alignment.
We release part of the dataset for training and analysis, the dataset is hosted on Hugging Face, click 👉 [here](https://huggingface.co/datasets/ZeyuJiang1/OrionEditBench). 

To reduce training overhead, some of multiple reference images are pre-composed into a single input (optionally with background removal) so they share a unified branch; examples are shown below (from left to right: reference image(s), source image, and synthesized result).

![The example of dataset.](assets/dataset.png)



## 📄 Disclaimer

This repository is built upon [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and is released under the Apache 2.0 License. We thank [Magiclight.AI](https://magiclight.ai) for their support in dataset collection and training resources. This project is intended for academic research and the broader AIGC community. Most of the released images are AI-generated or sourced from public datasets. For any concerns, please contact us; we will promptly review and remove inappropriate content.




## ⭐ Citation

If OrionEdit inspires your research 🤔, please consider giving this repo a ⭐ and citing our work:

```bibtex
@article{
> Our paper has been accepted to CVPR 2026 main track. 
The official citation will be released upon publication.
}