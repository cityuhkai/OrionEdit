# This file is adapted from the Hugging Face diffusers official Qwen-Image-Edit training example,
# aligned with diffusers version 0.36.0.dev0 and the Qwen-Image-Edit model family.
#
# For research and educational purposes only. For any commercial use, please comply with the original licenses and copyright terms of both Qwen-Image and diffusers.
# Copyright (c) Original Authors of diffusers and Qwen-Image-Edit. All rights reserved.
#
# Documentation:
# OrionEdit training entry point. The paper originally used Qwen-Image-Edit-2509; this repository trains against Qwen-Image-Edit-2511.
# The script extends the stock diffusers training loop with OrionEdit-specific components (e.g., CustomDataset, OrionEditPipeline,
# OrionEditTransformer2DModel, decoupling / LoRA helpers, and checkpoint utilities) while retaining diffusers schedulers, VAE hooks,
# optimization helpers, and accelerate-based distributed training patterns.
# For implementation details, refer to inline comments, argparse help strings, and the docstrings of the OrionEdit modules imported below.
# This file is intended for research and educational purposes only; for any commercial use, please comply with the original licenses of Qwen-Image and diffusers.

import argparse
import copy
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

# added
from safetensors.torch import load_file
import json
from models.custom_dataset import CustomDataset
from models.pipeline_orion_edit import OrionEditPipeline, build_dns_seg_slices
from models.transformer_orion import OrionEditTransformer2DModel
from models.transformer_orion import (
    set_orion_block_processors,
    enable_decoupling_trainables,
    get_decoupling_named_params,
)
from typing import Dict

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from models.utils import save_checkpoint
from safetensors.torch import save_file as save_sft
from accelerate.utils import DistributedDataParallelKwargs

from pathlib import Path

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log.get("validation_prompt", "")
            validation_image = log.get("validation_image")
            if validation_image is not None:
                validation_image.save(os.path.join(repo_folder, "image_control.png"))
                img_str += f"prompt: {validation_prompt}\n"
                images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# control-lora-{repo_id}

These are Control LoRA weights trained on {base_model} with new type of conditioning.
{img_str}

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "flux",
        "flux-diffusers",
        "text-to-image",
        "diffusers",
        "control-lora",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    import config.base as cfg

    defaults = dict(cfg.TRAIN_DEFAULTS)
    weighting_choices = list(getattr(cfg, "WEIGHTING_SCHEME_CHOICES", ("sigma_sqrt", "logit_normal", "mode", "cosmap", "none")))
    parser = argparse.ArgumentParser(
        description="OrionEdit Qwen training. Defaults: config/base.py; CLI only overrides set flags."
    )
    parser.set_defaults(**defaults)
    A = argparse.SUPPRESS

    for k in [
        "pretrained_model_name_or_path",
        "output_dir",
        "cache_dir",
        "resume_from_checkpoint",
        "lora_layers",
        "logging_dir",
        "report_to",
        "mixed_precision",
        "dataset_name",
        "dataset_config_name",
        "image_column",
        "conditioning_image_column",
        "caption_column",
        "tracker_project_name",
        "jsonl_for_train",
        "train_metadata_json",
        "wandb_run_name",
        "run_name",
        "hub_token",
        "hub_model_id",
        "lr_scheduler",
    ]:
        parser.add_argument(f"--{k.replace('_', '-')}", type=str, default=A)

    parser.add_argument("--weighting-scheme", type=str, choices=weighting_choices, default=A)

    for k in [
        "seed",
        "resolution",
        "train_batch_size",
        "num_train_epochs",
        "max_train_steps",
        "checkpointing_steps",
        "checkpoints_total_limit",
        "rank",
        "gradient_accumulation_steps",
        "lr_warmup_steps",
        "lr_num_cycles",
        "dataloader_num_workers",
        "max_train_samples",
        "inference_log_every",
        "inference_num_steps",
    ]:
        parser.add_argument(f"--{k.replace('_', '-')}", type=int, default=A)

    for k in [
        "proportion_empty_prompts",
        "learning_rate",
        "lr_power",
        "adam_beta1",
        "adam_beta2",
        "adam_weight_decay",
        "adam_epsilon",
        "max_grad_norm",
        "guidance_scale",
        "logit_mean",
        "logit_std",
        "mode_scale",
    ]:
        parser.add_argument(f"--{k.replace('_', '-')}", type=float, default=A)

    for k in [
        "push_to_hub",
        "scale_lr",
        "allow_tf32",
        "use_lora_bias",
        "gaussian_init_lora",
        "train_norm_layers",
        "upcast_before_saving",
        "log_dataset_samples",
        "from_tuned_checkpoint",
    ]:
        parser.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=A)

    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        default=A,
    )
    parser.add_argument("--no-8bit-adam", dest="use_8bit_adam", action="store_false", default=A)
    parser.add_argument("--no-offload", dest="offload", action="store_false", default=A)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    n_src = sum(
        [
            1 if args.train_metadata_json else 0,
            1 if args.dataset_name else 0,
            1 if args.jsonl_for_train else 0,
        ]
    )
    if n_src != 1:
        raise ValueError(
            "Specify exactly one of `--train_metadata_json`, `--dataset_name`, or `--jsonl_for_train`."
        )

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the Flux transformer."
        )

    return args


def get_train_dataset(args, accelerator):
    dataset = None
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    if args.jsonl_for_train is not None:
        # load from json
        dataset = load_dataset("json", data_files=args.jsonl_for_train, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices()
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [example["captions"] for example in examples]
    return {"pixel_values": pixel_values, "conditioning_pixel_values": conditioning_pixel_values, "captions": captions}


def collate_orion_metadata_batch(examples):
    """Preserves variable-length ``reference_image`` lists (fusion vs edit) per sample."""
    return {
        "prompt": [e["prompt"] for e in examples],
        "reference_image": [e["reference_image"] for e in examples],
        "source_image": [e["source_image"] for e in examples],
        "image": [e["image"] for e in examples],
    }


def _list_training_checkpoints(output_dir: str):
    """Return sorted list of (step_int, path) for ``checkpoints/step-*``, ``step-final-*``, or legacy ``checkpoint-*``."""
    found = []
    ckpt_root = os.path.join(output_dir, "checkpoints")
    if os.path.isdir(ckpt_root):
        for name in os.listdir(ckpt_root):
            p = os.path.join(ckpt_root, name)
            if not os.path.isdir(p):
                continue
            if name.startswith("step-final-"):
                try:
                    step = int(name[len("step-final-") :])
                    found.append((step, p))
                except ValueError:
                    pass
            elif name.startswith("step-"):
                try:
                    step = int(name[len("step-") :])
                    found.append((step, p))
                except ValueError:
                    pass
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                    found.append((step, os.path.join(output_dir, name)))
                except ValueError:
                    pass
    return sorted(found, key=lambda x: x[0])


@torch.no_grad()
def run_orion_inference_log(
    accelerator,
    args,
    weight_dtype,
    transformer,
    global_step: int,
    training_aux_pipelines=None,
):
    """Save preview images under ``output_dir/inference_log/step_{global_step}/``; return W&B image list."""
    if not accelerator.is_main_process:
        return []
    samples = getattr(args, "inference_log_samples", None) or []
    if not samples:
        return []

    training_aux_pipelines = training_aux_pipelines or []

    log_dir = os.path.join(args.output_dir, "inference_log", f"step_{global_step}")
    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped = unwrapped._orig_mod if is_compiled_module(unwrapped) else unwrapped
    was_training = unwrapped.training
    unwrapped.eval()
    wandb_images = []
    infer_pipe = None
    try:
        for _aux in training_aux_pipelines:
            _aux.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Do not call .to(cuda) here: training already holds VAE + text encoders + DDP transformer on GPU.
        # A second full pipeline duplicates them and OOMs (~80GB). Sequential offload runs one module on GPU at a time.
        infer_pipe = OrionEditPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=unwrapped,
            torch_dtype=weight_dtype,
        )
        # Shared training transformer must stay on real CUDA weights; offloading it breaks LoRA/meta tensors.
        _excl = list(getattr(infer_pipe, "_exclude_from_cpu_offload", None) or [])
        if "transformer" not in _excl:
            infer_pipe._exclude_from_cpu_offload = _excl + ["transformer"]
        infer_pipe.enable_sequential_cpu_offload(device=accelerator.device)
        os.makedirs(log_dir, exist_ok=True)

        for i, sample in enumerate(samples):
            prompt = sample.get("edit_prompt") or sample.get("prompt", "")
            ref = sample["reference_image"]
            src = sample.get("source_image", "") or ""
            if isinstance(ref, list):
                ref_in = [Image.open(p).convert("RGB") for p in ref]
            else:
                ref_in = Image.open(ref).convert("RGB")
            src_in = Image.open(src).convert("RGB") if str(src).strip() else None

            try:
                with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype, enabled=True):
                    out = infer_pipe(
                        prompt=prompt,
                        reference_image=ref_in,
                        source_image=src_in,
                        num_inference_steps=args.inference_num_steps,
                        true_cfg_scale=4.0,
                        negative_prompt=" ",
                        guidance_scale=args.guidance_scale,
                    )
            except (torch.OutOfMemoryError, RuntimeError) as infer_e:
                if isinstance(infer_e, RuntimeError) and "meta" not in str(infer_e).lower():
                    raise
                logger.warning(
                    "Inference log failed at step %s (sample %s): %s",
                    global_step,
                    i,
                    infer_e,
                )
                torch.cuda.empty_cache()
                continue
            pil_img = out.images[0]
            pil_img.save(os.path.join(log_dir, f"sample_{i:02d}.png"))
            caption = f"step={global_step} | {prompt[:200]}"
            if is_wandb_available() and args.report_to == "wandb":
                import wandb

                wandb_images.append(wandb.Image(pil_img, caption=caption))
    finally:
        unwrapped.train(was_training)
        if infer_pipe is not None:
            try:
                infer_pipe.remove_all_hooks()
            except Exception:
                pass
            del infer_pipe
        free_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            unwrapped.to(device=accelerator.device, dtype=weight_dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "Post-inference-log unwrapped.to() OOM (training may still be ok if weights stayed on GPU): %s",
                    e,
                )
            else:
                raise
        for _aux in training_aux_pipelines:
            try:
                _aux.to(accelerator.device)
            except RuntimeError as e:
                logger.warning("Could not move training aux pipeline back to %s: %s", accelerator.device, e)

    logger.info("Saved inference log for step %s to %s", global_step, log_dir)
    return wandb_images


def main(args):
    try:
        import config.base as _cfg_mod

        args.inference_log_samples = list(getattr(_cfg_mod, "INFERENCE_LOG_SAMPLES", []))
    except ImportError:
        args.inference_log_samples = []

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.use_lora_bias and args.gaussian_init_lora:
        raise ValueError("`gaussian` LoRA init scheme isn't supported when `use_lora_bias` is True.")

    logging_out_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))
    

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,           # 关键：允许本步中未参与反传的参数
        gradient_as_bucket_view=True,          # 更高效的 bucket 视图（可选）
        # static_graph=False,                  # 若你模型图是动态的，保持 False（默认）
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
        # Newer Accelerate: sampler seeding via DataLoaderConfiguration (not use_seedable_sampler= on Accelerator).
        # Avoids per-iter GPU broadcast of Generator state -> Invalid mt19937 state on multi-GPU.
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    
    print("load flux models...")
    
    #  TODO 普通加载
    qwen_transformer = OrionEditTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    
    # d = heads * head_dim（你的 block 内部通道数）
    set_orion_block_processors(qwen_transformer, d=3072, rank=256, use_gate=False)

    
    logger.info("All models loaded successfully")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    qwen_transformer.requires_grad_(False)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # let's not move the VAE to the GPU yet.
    qwen_transformer.to(dtype=weight_dtype, device=accelerator.device)



    # TODO 查看权重替换是否成功
    # final_dict = {}
    # for name, param in qwen_transformer.named_parameters():
    #     final_dict[name] = param


    if args.train_norm_layers:
        for name, param in qwen_transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj", # TODO 语义相关的不用训练
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            
            # TODO 新增
            "norm1.linear", # 用在double stream
            "norm1_context.linear",
            
            "norm.linear",
            "proj_mlp",
            "proj_out",
        ]
    
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    qwen_transformer.add_adapter(transformer_lora_config)

    # Unfreeze decoupling-specific trainables (low-rank projector, gates, knobs)
    enable_decoupling_trainables(qwen_transformer)

    # TODO
    # load fine tuned weights
    if args.from_tuned_checkpoint:
            
        state_dict_to_load = {}
        lora_dict = load_file("/data/zeyu/Flux_Kontext/output_0822_tmp_gpt_500_dataset/checkpoint-1000/pytorch_lora_weights.safetensors")
        for k in lora_dict.keys():
            new_k = ""
            new_k = k.replace("weight","default.weight")
            new_k = new_k.replace("transformer.","")
            state_dict_to_load[new_k] = lora_dict[k]
            
        qwen_transformer.load_state_dict(state_dict_to_load, strict=False)
        print("weights load sucessfull!")


    # TODO 打印所有transformer层
    print("check all trainable parameters...")
    for name, param in qwen_transformer.named_parameters():
        # if param.requires_grad == True: 
        print(f"Parameter {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        # if "A1" in name:
        #     print(f"Parameter {name}: shape={param.shape}, requires_grad={param.requires_grad}, param={param}")
            


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            print("------------------------going to accelerate save_model_hook------------------------")
            if accelerator.is_main_process:
                lora_only_sd = None
                extras_sd = {}

                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(qwen_transformer))):
                        model = unwrap_model(model)

                        # 1) 只取 LoRA（PEFT）权重 —— 这部分给 save_lora_weights
                        lora_only_sd = get_peft_model_state_dict(model)

                        # 2) 取自定义 Decoupled processor 的可训练权重 —— 这部分单独存到 processor 文件
                        proc_sd = {f"transformer.{k}": v for k, v in get_decoupling_named_params(model).items()}
                        extras_sd.update(proc_sd)

                        # 3) 若你训练了 norm 层，也把它们放到「extras」文件（不要进 LoRA 文件）
                        if args.train_norm_layers:
                            norm_sd = {
                                f"transformer.{name}": param
                                for name, param in model.named_parameters()
                                if any(k in name for k in NORM_LAYER_PREFIXES)
                            }
                            extras_sd.update(norm_sd)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # 防止 accelerate 再重复保存
                    if weights:
                        weights.pop()

                # === A. 保存 LoRA-only ===
                OrionEditPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=lora_only_sd,  # 只传 LoRA 键
                )

                # === B. 保存 extras（processor + 可选 norm）到另外一个 safetensors ===
                if len(extras_sd) > 0:
                    os.makedirs(output_dir, exist_ok=True)
                    save_sft(extras_sd, os.path.join(output_dir, "pytorch_processor_weights.safetensors"))


        def load_model_hook(models, input_dir):
            print("------------------------going to accelerate load_model_hook------------------------")
            transformer_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(unwrap_model(qwen_transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                transformer_ = OrionEditTransformer2DModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                ).to(accelerator.device, weight_dtype)

                # Handle input dimension doubling before adding adapter
                with torch.no_grad():
                    initial_input_channels = transformer_.config.in_channels
                    new_linear = torch.nn.Linear(
                        transformer_.x_embedder.in_features * 2,
                        transformer_.x_embedder.out_features,
                        bias=transformer_.x_embedder.bias is not None,
                        dtype=transformer_.dtype,
                        device=transformer_.device,
                    )
                    new_linear.weight.zero_()
                    new_linear.weight[:, :initial_input_channels].copy_(transformer_.x_embedder.weight)
                    if transformer_.x_embedder.bias is not None:
                        new_linear.bias.copy_(transformer_.x_embedder.bias)
                    transformer_.x_embedder = new_linear
                    transformer_.register_to_config(in_channels=initial_input_channels * 2)

                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = OrionEditPipeline.lora_state_dict(input_dir)
            transformer_lora_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and "lora" in k
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_lora_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            # load decoupling processor parameters if present
            dec_sd = {k.replace('transformer.', ''): v for k, v in lora_state_dict.items() if 'attn.processor' in k}
            if dec_sd:
                missing, unexpected = transformer_.load_state_dict(dec_sd, strict=False)
                if missing or unexpected:
                    logger.info(f"Loaded processor params with missing={missing}, unexpected={unexpected}")

            if args.train_norm_layers:
                transformer_norm_state_dict = {
                    k: v
                    for k, v in lora_state_dict.items()
                    if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
                }
                transformer_._transformer_norm_layers = OrionEditPipeline._load_norm_into_transformer(
                    transformer_norm_state_dict,
                    transformer=transformer_,
                    discard_original_layers=False,
                )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16":
                models = [transformer_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)



    # TODO 确保trainable的parameters是float32的
    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [qwen_transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.gradient_checkpointing:
        qwen_transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimization parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, qwen_transformer.parameters()))
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print("preparing datasets...")

    if args.train_metadata_json:
        train_dataset = CustomDataset(
            data_meta_paths=[args.train_metadata_json],
            shuffle_seed=args.seed,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collate_orion_metadata_batch,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_dataset = get_train_dataset(args, accelerator)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )


    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    print("preparing accelerator...")
    
    # Prepare everything with our `accelerator`.
    qwen_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        qwen_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Trackers (W&B / TensorBoard): config must be JSON-serializable.
    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = dict(vars(args))
        for _k, _v in list(tracker_config.items()):
            if _v is None:
                tracker_config.pop(_k, None)
        tracker_config.pop("inference_log_samples", None)
        init_kwargs = None
        if args.report_to == "wandb" and is_wandb_available():
            run_display = args.run_name or args.wandb_run_name or Path(args.output_dir).name
            init_kwargs = {"wandb": {"name": run_display}}
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
    flux_qwen_vae_pipeline = OrionEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path, scheduler=None, transformer=None, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, torch_dtype=torch.float32
    )
    flux_qwen_encoder_pipeline = OrionEditPipeline.from_pretrained(args.pretrained_model_name_or_path, vae=None, transformer=None,  torch_dtype=weight_dtype)
    vae_scale_factor = flux_qwen_vae_pipeline.vae_scale_factor * 2
    flux_qwen_vae_pipeline = flux_qwen_vae_pipeline.to(accelerator.device)
    flux_qwen_encoder_pipeline = flux_qwen_encoder_pipeline.to(accelerator.device)

    def _force_override_lr(optimizer_obj, scheduler_obj, target_lr: float):
        """Force optimizer/scheduler LR to target value (including Accelerate wrappers)."""
        # 1) Optimizer param groups
        for group in optimizer_obj.param_groups:
            group["lr"] = target_lr
            if "initial_lr" in group:
                group["initial_lr"] = target_lr

        # 2) Scheduler (handle AcceleratedScheduler wrapper)
        sched = scheduler_obj.scheduler if hasattr(scheduler_obj, "scheduler") else scheduler_obj
        if hasattr(sched, "base_lrs"):
            sched.base_lrs = [target_lr for _ in sched.base_lrs]
        if hasattr(sched, "_last_lr"):
            sched._last_lr = [target_lr for _ in sched._last_lr]
        if hasattr(sched, "optimizer") and getattr(sched.optimizer, "param_groups", None):
            for group in sched.optimizer.param_groups:
                group["lr"] = target_lr
                if "initial_lr" in group:
                    group["initial_lr"] = target_lr

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            cand = args.resume_from_checkpoint
            load_dir = cand if os.path.isdir(cand) else os.path.join(args.output_dir, os.path.basename(cand.rstrip("/")))
        else:
            found_ckpts = _list_training_checkpoints(args.output_dir)
            load_dir = found_ckpts[-1][1] if found_ckpts else None

        if load_dir is None or not os.path.isdir(load_dir):
            logger.info(
                "Checkpoint '%s' not found or invalid. Starting a new training run.",
                args.resume_from_checkpoint,
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info("Resuming from %s", load_dir)
            accelerator.load_state(load_dir)
            base = os.path.basename(load_dir.rstrip("/"))
            global_step = 0
            if base.startswith("step-final-"):
                try:
                    global_step = int(base[len("step-final-") :])
                except ValueError:
                    global_step = 0
            else:
                for prefix in ("step-", "checkpoint-"):
                    if base.startswith(prefix):
                        try:
                            global_step = int(base[len(prefix) :])
                        except ValueError:
                            global_step = 0
                        break
            # `accelerator.load_state` restores optimizer/scheduler states from checkpoint,
            # which also restores the old LR. Re-apply current config LR after resume.
            resumed_lr = None
            if getattr(optimizer, "param_groups", None):
                resumed_lr = optimizer.param_groups[0].get("lr", None)
            if resumed_lr is not None and not math.isclose(resumed_lr, args.learning_rate, rel_tol=0.0, abs_tol=1e-12):
                logger.info(
                    "Overriding resumed learning rate from %.8g to configured %.8g",
                    resumed_lr,
                    args.learning_rate,
                )
            _force_override_lr(optimizer, lr_scheduler, args.learning_rate)
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Inference preview at step 0 (before any optimizer step) for visual baseline
    if (
        initial_global_step == 0
        and args.train_metadata_json
        and getattr(args, "inference_log_samples", None)
    ):
        pre_imgs = run_orion_inference_log(
            accelerator,
            args,
            weight_dtype,
            qwen_transformer,
            0,
            training_aux_pipelines=[flux_qwen_vae_pipeline, flux_qwen_encoder_pipeline],
        )
        if pre_imgs and accelerator.is_main_process and args.report_to != "none":
            accelerator.log({"inference_preview": pre_imgs, "train/step": 0}, step=0)
    # Non-main ranks must not start training while main runs step-0 inference (same collective order on all ranks).
    accelerator.wait_for_everyone()

    if (
        accelerator.is_main_process
        and args.report_to == "wandb"
        and args.log_dataset_samples
        and not args.train_metadata_json
    ):
        logger.info("Logging some dataset samples.")
        formatted_images = []
        formatted_control_images = []
        all_prompts = []
        for i, batch in enumerate(train_dataloader):
            images = (batch["pixel_values"] + 1) / 2
            control_images = (batch["conditioning_pixel_values"] + 1) / 2
            prompts = batch["captions"]

            if len(formatted_images) > 10:
                break

            for img, control_img, prompt in zip(images, control_images, prompts):
                formatted_images.append(img)
                formatted_control_images.append(control_img)
                all_prompts.append(prompt)

        if not is_wandb_available():
            logger.warning("log_dataset_samples skipped: wandb is not installed or unavailable.")
        else:
            import wandb

            logged_artifacts = []
            for img, control_img, prompt in zip(formatted_images, formatted_control_images, all_prompts):
                logged_artifacts.append(wandb.Image(control_img, caption="Conditioning"))
                logged_artifacts.append(wandb.Image(img, caption=prompt))

            wandb_tracker = [tracker for tracker in accelerator.trackers if tracker.name == "wandb"]
            wandb_tracker[0].log({"dataset_samples": logged_artifacts})

    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    print("going to train loop...")

    image_logs = None
    

    for epoch in range(first_epoch, args.num_train_epochs):
        qwen_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(qwen_transformer):
                num_images_per_prompt = 1

                def preprocess_image(image_path):
                    image = Image.open(image_path).convert("RGB")
                    calculated_width, calculated_height, _ = calculate_dimensions(
                        1024 * 1024, image.size[0] / image.size[1]
                    )

                    image = flux_qwen_vae_pipeline.image_processor.resize(
                        image, calculated_height, calculated_width
                    )
                    prompt_image = image
                    image = flux_qwen_vae_pipeline.image_processor.preprocess(
                        image, calculated_height, calculated_width
                    )
                    image = image.unsqueeze(2)
                    image = image.to(device=accelerator.device, dtype=torch.float32)

                    image_latents = flux_qwen_vae_pipeline._encode_vae_image(image=image, generator=None)
                    image_latents = image_latents.to(dtype=weight_dtype, device=accelerator.device)

                    return image_latents, prompt_image, calculated_width, calculated_height

                per_sample_losses = []
                n_in_batch = len(batch["prompt"])
                for bi in range(n_in_batch):
                    prompt = batch["prompt"][bi]
                    ref_list = batch["reference_image"][bi]
                    src_s = batch["source_image"][bi]
                    target_path = batch["image"][bi]
                    has_source = bool(str(src_s or "").strip())

                    with torch.no_grad():
                        image_latents, _, image_width, image_height = preprocess_image(target_path)
                        ref_entries = []
                        for rp in ref_list:
                            lat, pil_im, cw, ch = preprocess_image(rp)
                            ref_entries.append(
                                (
                                    lat.to(device=accelerator.device),
                                    pil_im,
                                    cw,
                                    ch,
                                )
                            )

                        if has_source:
                            src_latents, prompt_src, src_w, src_h = preprocess_image(src_s)
                            src_latents = src_latents.to(device=accelerator.device)
                        else:
                            src_latents = None
                            prompt_src = None
                            src_w = src_h = None

                        image_latents = image_latents.to(device=accelerator.device)
                        condition_images = [e[1] for e in ref_entries]
                        if has_source:
                            condition_images.append(prompt_src)

                        prompt_embeds, prompt_embeds_mask, _ = flux_qwen_encoder_pipeline.encode_prompt(
                            image=condition_images,
                            prompt=prompt,
                            prompt_embeds=None,
                            prompt_embeds_mask=None,
                            device=accelerator.device,
                            num_images_per_prompt=num_images_per_prompt,
                            max_sequence_length=1024,
                        )
                        txt_seq_lens = (
                            prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
                        )

                        bsz = image_latents.shape[0]

                        packed_refs = [
                            OrionEditPipeline._pack_latents(
                                e[0],
                                batch_size=bsz,
                                num_channels_latents=e[0].shape[1],
                                height=e[0].shape[3],
                                width=e[0].shape[4],
                            )
                            for e in ref_entries
                        ]

                        img_shape_row = [
                            (1, image_height // vae_scale_factor, image_width // vae_scale_factor),
                        ]
                        for _, _, ch, cw in ref_entries:
                            img_shape_row.append((1, ch // vae_scale_factor, cw // vae_scale_factor))
                        if has_source:
                            img_shape_row.append((1, src_h // vae_scale_factor, src_w // vae_scale_factor))

                        img_shapes = [img_shape_row] * bsz

                        if has_source:
                            packed_src = OrionEditPipeline._pack_latents(
                                src_latents,
                                batch_size=bsz,
                                num_channels_latents=src_latents.shape[1],
                                height=src_latents.shape[3],
                                width=src_latents.shape[4],
                            )
                            latent_cond = torch.cat([packed_refs[0], packed_src], dim=1)
                            packed_lengths_for_seg = [packed_refs[0].shape[1], packed_src.shape[1]]
                        else:
                            latent_cond = torch.cat(packed_refs, dim=1)
                            packed_lengths_for_seg = [p.shape[1] for p in packed_refs]

                    noise = torch.randn_like(image_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                    sigmas = get_sigmas(timesteps, n_dim=image_latents.ndim, dtype=weight_dtype)
                    noisy_model_input = (1.0 - sigmas) * image_latents + sigmas * noise

                    packed_noisy_model_input = OrionEditPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=bsz,
                        num_channels_latents=noisy_model_input.shape[1],
                        height=noisy_model_input.shape[3],
                        width=noisy_model_input.shape[4],
                    )

                    if unwrap_model(qwen_transformer).config.guidance_embeds:
                        guidance_vec = torch.full(
                            (bsz,),
                            args.guidance_scale,
                            device=noisy_model_input.device,
                            dtype=weight_dtype,
                        )
                    else:
                        guidance_vec = None

                    if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                        prompt_embeds.zero_()

                    latent_model_input = torch.cat([packed_noisy_model_input, latent_cond], dim=1)

                    seg_slices_img = build_dns_seg_slices(
                        packed_noisy_model_input.shape[1],
                        packed_lengths_for_seg,
                        has_source=has_source,
                    )
                    _joint_attention_kwargs = {
                        "seg_slices_img": seg_slices_img,
                        "soft_beta": 6.0,
                    }

                    _xf_kw = {}
                    if getattr(unwrap_model(qwen_transformer).config, "use_additional_t_cond", False):
                        _xf_kw["additional_t_cond"] = torch.zeros(
                            bsz, dtype=torch.long, device=accelerator.device
                        )

                    model_pred = qwen_transformer(
                        hidden_states=latent_model_input,
                        timestep=timesteps / 1000,
                        guidance=guidance_vec,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=_joint_attention_kwargs,
                        return_dict=False,
                        **_xf_kw,
                    )[0]
                    model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                    model_pred = OrionEditPipeline._unpack_latents(
                        model_pred,
                        height=noisy_model_input.shape[3] * vae_scale_factor,
                        width=noisy_model_input.shape[4] * vae_scale_factor,
                        vae_scale_factor=vae_scale_factor,
                    )
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                    target = noise - image_latents
                    loss_i = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                            target.shape[0], -1
                        ),
                        1,
                    ).mean()
                    per_sample_losses.append(loss_i)

                loss = torch.stack(per_sample_losses).mean()
                accelerator.backward(loss)
                
                # TODO 查看梯度
                # for n, p in qwen_transformer.named_parameters():
                    # if "attn.processor.B1" in n:
                    #     g = None if p.grad is None else p.grad.norm().item()
                    #     print("[GRAD CHECK]", n, "value=", p, "grad_none=", p.grad is None, "grad_norm=", g)
                    #     break

                if accelerator.sync_gradients:
                    params_to_clip = qwen_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # Resume path may restore old scheduler internals; enforce configured LR.
                if args.resume_from_checkpoint and args.lr_scheduler == "constant":
                    _force_override_lr(optimizer, lr_scheduler, args.learning_rate)
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                preview_images = []
                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        ckpt_root = os.path.join(args.output_dir, "checkpoints")
                        if args.checkpoints_total_limit is not None and os.path.isdir(ckpt_root):
                            step_entries = []
                            for name in os.listdir(ckpt_root):
                                p = os.path.join(ckpt_root, name)
                                if not os.path.isdir(p):
                                    continue
                                if name.startswith("step-final-"):
                                    try:
                                        step_entries.append((int(name[len("step-final-") :]), name))
                                    except ValueError:
                                        pass
                                elif name.startswith("step-"):
                                    try:
                                        step_entries.append((int(name[len("step-") :]), name))
                                    except ValueError:
                                        pass
                            step_entries.sort(key=lambda x: x[0])
                            step_dirs = [n for _, n in step_entries]
                            while len(step_dirs) >= args.checkpoints_total_limit:
                                oldest = step_dirs.pop(0)
                                shutil.rmtree(os.path.join(ckpt_root, oldest), ignore_errors=True)

                        save_path = os.path.join(args.output_dir, "checkpoints", f"step-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        logger.info("Saved state to %s (LoRA + processor safetensors via save hook)", save_path)

                    if (
                        args.train_metadata_json
                        and args.inference_log_every > 0
                        and global_step % args.inference_log_every == 0
                        and getattr(args, "inference_log_samples", None)
                    ):
                        preview_images = run_orion_inference_log(
                            accelerator,
                            args,
                            weight_dtype,
                            qwen_transformer,
                            global_step,
                            training_aux_pipelines=[flux_qwen_vae_pipeline, flux_qwen_encoder_pipeline],
                        )

                # Main may spend minutes in checkpoint + inference_log; workers must wait or they enter the next
                # forward/backward and hang in ALLREDUCE while rank0 is still in inference (NCCL timeout ~600s).
                accelerator.wait_for_everyone()

                if accelerator.is_main_process and args.report_to != "none":
                    logs = {
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    }
                    if preview_images:
                        logs["inference_preview"] = preview_images
                    accelerator.log(logs, step=global_step)

            progress_bar.set_postfix(
                loss=loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
            )

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        qwen_transformer = unwrap_model(qwen_transformer)
        if args.upcast_before_saving:
            qwen_transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(qwen_transformer)
        # also save decoupling processor trainables
        extra_proc = {f"transformer.{k}": v for k, v in get_decoupling_named_params(qwen_transformer).items()}
        transformer_lora_layers = {**transformer_lora_layers, **extra_proc}
        if args.train_norm_layers:
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in qwen_transformer.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            transformer_lora_layers = {**transformer_lora_layers, **transformer_norm_layers}

        if args.train_metadata_json and getattr(args, "inference_log_samples", None):
            final_imgs = run_orion_inference_log(
                accelerator,
                args,
                weight_dtype,
                qwen_transformer,
                global_step,
                training_aux_pipelines=[flux_qwen_vae_pipeline, flux_qwen_encoder_pipeline],
            )
            if final_imgs and args.report_to != "none":
                accelerator.log(
                    {"inference_preview_final": final_imgs, "train/step": global_step},
                    step=global_step,
                )

        final_ckpt = os.path.join(args.output_dir, "checkpoints", f"step-final-{global_step}")
        os.makedirs(final_ckpt, exist_ok=True)
        lora_only_sd = get_peft_model_state_dict(qwen_transformer)
        OrionEditPipeline.save_lora_weights(
            save_directory=final_ckpt,
            transformer_lora_layers=lora_only_sd,
        )
        extras_sd = {f"transformer.{k}": v for k, v in get_decoupling_named_params(qwen_transformer).items()}
        if args.train_norm_layers:
            norm_sd = {
                f"transformer.{name}": param
                for name, param in qwen_transformer.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            extras_sd.update(norm_sd)
        if extras_sd:
            save_sft(extras_sd, os.path.join(final_ckpt, "pytorch_processor_weights.safetensors"))

        OrionEditPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        del qwen_transformer
        del flux_qwen_vae_pipeline
        del flux_qwen_encoder_pipeline
        free_memory()

        image_logs = None

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "*.pt", "*.bin"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    main(args)



