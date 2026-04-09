# Legacy fields that are still supported for backward compatibility, but not used in the new pipeline format:
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import os
import torch
from torch import Tensor
from diffusers.pipelines import FluxPipeline
from diffusers.utils import logging
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.pipelines.flux.pipeline_flux import FluxLoraLoaderMixin
from diffusers.models.transformers.transformer_flux import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)
from peft.tuners.tuners_utils import BaseTunerLayer
# from optimum.quanto import (
#     freeze, quantize, QTensor, qfloat8, qint8, qint4, qint2,
# )
import re
import safetensors
# from src.adapters.mod_adapters import CLIPModAdapter
from peft import LoraConfig, set_peft_model_state_dict
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection, CLIPVisionModel


def encode_vae_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids

def decode_vae_images(pipeline: FluxPipeline, latents: Tensor, height, width, output_type: Optional[str] = "pil"):
    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type=output_type)


def _get_clip_prompt_embeds(
    self,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if isinstance(self, TextualInversionLoaderMixin):
        prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids

    prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt_with_clip_t5(
    self,
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
):
    r"""

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in all text-encoders
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        lora_scale (`float`, *optional*):
            A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
    """
    device = device or self._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
        self._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if self.text_encoder is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = _get_clip_prompt_embeds(
            self=self,
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if self.text_encoder_2 is not None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            
    if self.text_encoder is not None:
        if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

    if self.text_encoder_2 is not None:
        if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder_2, lora_scale)

    dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
    if self.text_encoder_2 is not None:
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    else:
        text_ids = None

    return prompt_embeds, pooled_prompt_embeds, text_ids



def prepare_text_input(
    pipeline: FluxPipeline, 
    prompts, 
    max_sequence_length=512,
):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        t5_prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = encode_prompt_with_clip_t5(
        self=pipeline,
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return t5_prompt_embeds, pooled_prompt_embeds, text_ids

def prepare_t5_input(
    pipeline: FluxPipeline, 
    prompts, 
    max_sequence_length=512,
):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        t5_prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = encode_prompt_with_clip_t5(
        self=pipeline,
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return t5_prompt_embeds, pooled_prompt_embeds, text_ids

def tokenize_t5_prompt(pipe, input_prompt, max_length, **kargs):
    return pipe.tokenizer_2(
        input_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
        **kargs,
    )

def quantization(pipe, qtype):
    if qtype != "None" and qtype != "":
        if qtype.endswith("quanto"):
            if qtype == "int2-quanto":
                quant_level = qint2
            elif qtype == "int4-quanto":
                quant_level = qint4
            elif qtype == "int8-quanto":
                quant_level = qint8
            elif qtype == "fp8-quanto":
                quant_level = qfloat8
            else:
                raise ValueError(f"Invalid quantisation level: {qtype}")

            extra_quanto_args = {}
            extra_quanto_args["exclude"] = [
                "*.norm",
                "*.norm1",
                "*.norm2",
                "*.norm2_context",
                "proj_out",
                "x_embedder",
                "norm_out",
                "context_embedder",
            ]
            try:
                print("[Quantization] Start freezing")
                quantize(pipe.transformer, weights=quant_level, **extra_quanto_args)
                freeze(pipe.transformer)
                quantize(pipe.text_encoder_2, weights=quant_level, **extra_quanto_args)
                freeze(pipe.text_encoder_2)
                torch.cuda.empty_cache()
                print("[Quantization] Finished")
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(
                        "GPU ran out of memory during quantisation. Use --quantize_via=cpu to use the slower CPU method."
                    )
                raise e
        else:
            assert qtype == "fp8-ao"
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            def module_filter_fn(mod: torch.nn.Module, fqn: str):
                # don't convert the output module
                if fqn == "proj_out":
                    return False
                # don't convert linear modules with weight dimensions not divisible by 16
                if isinstance(mod, torch.nn.Linear):
                    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                        return False
                return True
            convert_to_float8_training(
                pipe.transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
            )

class CustomFluxPipeline:
    def __init__(
        self,
        config,
        device=torch.device("cuda"),
        ckpt_root=None,
        ckpt_root_condition=None,
        torch_dtype=torch.bfloat16,
    ):
        # model_path = os.getenv("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
        model_path = os.getenv("/data/zeyu/XVerse-main/checkpoints/FLUX.1-dev", "/data/zeyu/XVerse-main/checkpoints/FLUX.1-dev")
        print("[CustomFluxPipeline] Loading FLUX Pipeline")
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

        # self.pipe.vae = self.pipe.vae.to(device, dtype=torch_dtype)
        # self.pipe.transformer = self.pipe.transformer.to(device, dtype=torch_dtype)

        self.config = config
        self.device = device
        self.dtype = torch_dtype
        start = time.time()
        if config["model"].get("dit_quant", "None") != "None":
            quantization(self.pipe, config["model"]["dit_quant"])
        print(f"Quantization time: {time.time() - start}")

        self.pipe = self.pipe.to(device)

        self.modulation_adapters = []
        self.pipe.modulation_adapters = []
        
        try:
            if config["model"]["modulation"]["use_clip"]:
                load_clip(self, config, torch_dtype, device, None, is_training=False)
        except Exception as e:
            print(e)
        
        if config["model"]["use_dit_lora"] or config["model"]["use_condition_dblock_lora"] or config["model"]["use_condition_sblock_lora"]:
            if ckpt_root_condition is None and (config["model"]["use_condition_dblock_lora"] or config["model"]["use_condition_sblock_lora"]):
                ckpt_root_condition = ckpt_root
            load_dit_lora(self, self.pipe, config, torch_dtype, device, f"{ckpt_root}", f"{ckpt_root_condition}", is_training=False)
        self.pipe.to(device)

    def add_modulation_adapter(self, modulation_adapter):
        self.modulation_adapters.append(modulation_adapter)
        self.pipe.modulation_adapters.append(modulation_adapter)

    def clear_modulation_adapters(self):
        self.modulation_adapters = []
        self.pipe.modulation_adapters = []
        torch.cuda.empty_cache()

def load_clip(pipeline, config, torch_dtype, device, ckpt_dir=None, is_training=False):
    # model_path = os.getenv("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")
    model_path = os.getenv("CLIP_MODEL_PATH", "/data/zeyu/XVerse-main/checkpoints/clip-vit-large-patch14")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(device, dtype=torch_dtype)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    pipeline.pipe.clip_model = clip_model
    pipeline.pipe.clip_processor = clip_processor

def load_dit_lora(pipeline, pipe, config, torch_dtype, device, ckpt_dir=None, condition_ckpt_dir=None, is_training=False):

    if not config["model"]["use_condition_dblock_lora"] and not config["model"]["use_condition_sblock_lora"] and not config["model"]["use_dit_lora"]:
        print("[load_dit_lora] no dit lora, no condition lora")
        return []
    
    adapter_names = ["default", "condition"]
    
    if condition_ckpt_dir is None:
        condition_ckpt_dir = ckpt_dir
    
    if not config["model"]["use_condition_dblock_lora"] and not config["model"]["use_condition_sblock_lora"]:
        print("[load_dit_lora] no condition lora")
        adapter_names.pop(1)
    elif condition_ckpt_dir is not None and os.path.exists(os.path.join(condition_ckpt_dir, "pytorch_lora_weights_condition.safetensors")):
        assert "condition" in adapter_names
        print(f"[load_dit_lora] load condition lora from {condition_ckpt_dir}")
        pipe.transformer.load_lora_adapter(condition_ckpt_dir, use_safetensors=True, adapter_name="condition", weight_name="pytorch_lora_weights_condition.safetensors") # TODO: check if they are trainable
    else:
        assert is_training
        assert "condition" in adapter_names
        print("[load_dit_lora] init new condition lora")
        pipe.transformer.add_adapter(LoraConfig(**config["model"]["condition_lora_config"]), adapter_name="condition")

    if not config["model"]["use_dit_lora"]:
        print("[load_dit_lora] no dit lora")
        adapter_names.pop(0)
    elif ckpt_dir is not None and os.path.exists(os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")):
        assert "default" in adapter_names
        print(f"[load_dit_lora] load dit lora from {ckpt_dir}")
        lora_file = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")
        lora_state_dict = safetensors.torch.load_file(lora_file, device="cpu")
        
        single_lora_pattern = "(.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"
        latent_lora_pattern = "(.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2)"
        use_pretrained_dit_single_lora = config["model"].get("use_pretrained_dit_single_lora", True)
        use_pretrained_dit_latent_lora = config["model"].get("use_pretrained_dit_latent_lora", True)
        if not use_pretrained_dit_single_lora or not use_pretrained_dit_latent_lora:
            lora_state_dict_keys = list(lora_state_dict.keys())
            for layer_name in lora_state_dict_keys:
                if not use_pretrained_dit_single_lora:
                    if re.search(single_lora_pattern, layer_name):
                        del lora_state_dict[layer_name]
                if not use_pretrained_dit_latent_lora:
                    if re.search(latent_lora_pattern, layer_name):
                        del lora_state_dict[layer_name]
            pipe.transformer.add_adapter(LoraConfig(**config["model"]["dit_lora_config"]), adapter_name="default")
            set_peft_model_state_dict(pipe.transformer, lora_state_dict, adapter_name="default")
        else:
            pipe.transformer.load_lora_adapter(ckpt_dir, use_safetensors=True, adapter_name="default", weight_name="pytorch_lora_weights.safetensors") # TODO: check if they are trainable
    else:
        assert is_training
        assert "default" in adapter_names
        print("[load_dit_lora] init new dit lora")
        pipe.transformer.add_adapter(LoraConfig(**config["model"]["dit_lora_config"]), adapter_name="default")

    assert len(adapter_names) <= 2 and len(adapter_names) > 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, BaseTunerLayer):
            module.set_adapter(adapter_names)
    
    if "default" in adapter_names: assert config["model"]["use_dit_lora"]
    if "condition" in adapter_names: assert config["model"]["use_condition_dblock_lora"] or config["model"]["use_condition_sblock_lora"]
    
    lora_layers = list(filter(
        lambda p: p[1].requires_grad, pipe.transformer.named_parameters()
    ))
    
    lora_layers = [l[1] for l in lora_layers]
    return lora_layers

def load_modulation_adapter(pipeline, config, torch_dtype, device, ckpt_dir=None, is_training=False):
    adapter_type = config["model"]["modulation"]["adapter_type"]

    if ckpt_dir is not None and os.path.exists(ckpt_dir):
        print(f"loading modulation adapter from {ckpt_dir}")
        modulation_adapter = CLIPModAdapter.from_pretrained(
            ckpt_dir, subfolder="modulation_adapter", strict=False, 
            low_cpu_mem_usage=False, device_map=None,
        ).to(device)
    else:
        print(f"Init new modulation adapter")
        adapter_layers = config["model"]["modulation"]["adapter_layers"]
        adapter_width = config["model"]["modulation"]["adapter_width"]
        pblock_adapter_layers = config["model"]["modulation"]["per_block_adapter_layers"]
        pblock_adapter_width = config["model"]["modulation"]["per_block_adapter_width"]
        pblock_adapter_single_blocks = config["model"]["modulation"]["per_block_adapter_single_blocks"]
        use_text_mod = config["model"]["modulation"]["use_text_mod"]
        use_img_mod = config["model"]["modulation"]["use_img_mod"]
        
        out_dim = config["model"]["modulation"]["out_dim"]
        if adapter_type == "clip_adapter":
            modulation_adapter = CLIPModAdapter(
                out_dim=out_dim,
                width=adapter_width,
                pblock_width=pblock_adapter_width,
                layers=adapter_layers,
                pblock_layers=pblock_adapter_layers,
                heads=8,
                input_text_dim=4096,
                input_image_dim=1024,
                pblock_single_blocks=pblock_adapter_single_blocks,
            )
        else:
            raise NotImplementedError()

    if is_training:
        modulation_adapter.train()
        try:
            modulation_adapter.enable_gradient_checkpointing()
        except Exception as e:
            print(e)
        if not config["model"]["modulation"]["use_perblock_adapter"]:
            try:
                modulation_adapter.net2.requires_grad_(False)
            except Exception as e:
                print(e)
    else:
        modulation_adapter.requires_grad_(False)
    modulation_adapter.to(device, dtype=torch_dtype)
    return modulation_adapter


def load_ckpt(pipeline, ckpt_dir, is_training=False):
    if pipeline.config["model"]["use_dit_lora"]:
        pipeline.pipe.transformer.delete_adapters(["subject"])
        lora_path = f"{ckpt_dir}/pytorch_lora_weights.safetensors"
        print(f"Loading DIT Lora from {lora_path}")
        pipeline.pipe.load_lora_weights(lora_path, adapter_name="subject")

        

