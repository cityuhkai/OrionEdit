# This file is modified from diffusers' transformer_qwenimage.py, based on diffusers version 0.36.0.dev0.
# Qwen Image version: Qwen-image-edit-2511.
#
# For research and educational purposes only. For any commercial use, please comply with the original licenses and copyright terms of both Qwen-Image and diffusers.
# Copyright (c) Original Authors of diffusers and Qwen-image-edit. All rights reserved.

import functools
import math
from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm

try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn as _dispatch_attention_fn
except ImportError:  # older diffusers
    def _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=None,
        **kwargs,
    ):
        kwargs.pop("parallel_config", None)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        return out.transpose(1, 2).contiguous()


try:
    from diffusers.models.attention import AttentionMixin as _AttentionMixin
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    _HAS_ATTENTION_MIXIN = True
except ImportError:
    _AttentionMixin = None  # type: ignore
    ContextParallelInput = None  # type: ignore
    ContextParallelOutput = None  # type: ignore
    _HAS_ATTENTION_MIXIN = False


logger = logging.get_logger(__name__)

STAGE_A_LAYERS = 20
STAGE_B_LAYERS = 20

# joint_attention_kwargs / attention_kwargs: seg_slices_img must be {"syn","ref","src"} from
# build_image_stream_slices (order: noisy synthesis | reference | source). L_src==0 => fusion.
# Stage A/B: Orion-style reverse-causal mask on joint [text | syn | ref | src] (paper §3.2, Eq. 8).
#   G1 = text & syn: queries may attend all keys (read ref, src, text, syn).
#   G2 = src: may attend ref & src keys only (blocked from text & syn keys).
#   G3 = ref: ref keys only (blocked from text, syn, and src keys).
# Stage A uses -inf on blocked logits; Stage B subtracts soft_beta. StageB reads soft_beta from kwargs / proc.


def _dispatch_joint_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.FloatTensor],
    backend: Optional[str],
    parallel_config: Any = None,
) -> torch.Tensor:
    kwargs: Dict[str, Any] = {
        "attn_mask": attn_mask,
        "dropout_p": 0.0,
        "is_causal": False,
        "backend": backend,
    }
    if parallel_config is not None:
        kwargs["parallel_config"] = parallel_config
    try:
        return _dispatch_attention_fn(query, key, value, **kwargs)
    except TypeError:
        kwargs.pop("parallel_config", None)
        return _dispatch_attention_fn(query, key, value, **kwargs)


_DNS_MODEL_BASES: Tuple[type, ...] = (
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
)
if _HAS_ATTENTION_MIXIN:
    _DNS_MODEL_BASES = _DNS_MODEL_BASES + (_AttentionMixin,)  # type: ignore


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class OrionTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, use_additional_t_cond: bool = False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond: Optional[torch.Tensor] = None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When use_additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class OrionEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class OrionDoubleStreamAttnProcessor2_0:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "OrionDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("OrionDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = _dispatch_joint_attention(
            joint_query,
            joint_key,
            joint_value,
            attention_mask,
            self._attention_backend,
            getattr(self, "_parallel_config", None),
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class OrionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.zero_cond_t = zero_cond_t

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=OrionDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)

            shift_0_exp = shift_0.unsqueeze(1)
            shift_1_exp = shift_1.unsqueeze(1)
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        img_attn_output, txt_attn_output = attn_output

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class OrionEditTransformer2DModel(*_DNS_MODEL_BASES):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["OrionTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["OrionTransformerBlock"]

    if _HAS_ATTENTION_MIXIN and ContextParallelInput is not None and ContextParallelOutput is not None:
        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "encoder_hidden_states_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
            },
            "pos_embed": {
                0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
    else:
        _cp_plan = None  # type: ignore

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__()
        if use_layer3d_rope:
            raise NotImplementedError("use_layer3d_rope=True is not implemented; use the 2511 reference or set False.")

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = OrionEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = OrionTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                OrionTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    attention_kwargs,
                    modulate_index,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    modulate_index=modulate_index,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def build_image_stream_slices(L_syn: int, L_ref: int, L_src: int, start_offset: int = 0) -> Dict[str, slice]:
    """Image stream layout: syn | ref | src (packed latents left-to-right). `src` may be empty when L_src==0 (fusion)."""
    syn = slice(start_offset, start_offset + L_syn)
    ref = slice(start_offset + L_syn, start_offset + L_syn + L_ref)
    src = slice(start_offset + L_syn + L_ref, start_offset + L_syn + L_ref + L_src)
    return {"syn": syn, "ref": ref, "src": src}


def build_segment_slices_img(Lz: int, L1: int, L2: int, start_offset: int = 0) -> Dict[str, slice]:
    """Alias of build_image_stream_slices; returns keys syn, ref, src only."""
    return build_image_stream_slices(Lz, L1, L2, start_offset)


def _coerce_seg_slices(seg: Optional[Dict[str, slice]]) -> Dict[str, slice]:
    if not seg:
        return {}
    if "syn" in seg and "ref" in seg and "src" in seg:
        return {"syn": seg["syn"], "ref": seg["ref"], "src": seg["src"]}
    raise ValueError("seg_slices_img must contain keys syn, ref, and src (see build_image_stream_slices).")


def _joint_slices(seq_txt: int, seg: Dict[str, slice]) -> Tuple[slice, slice, slice]:
    s = _coerce_seg_slices(seg)
    syn = slice(seq_txt + s["syn"].start, seq_txt + s["syn"].stop)
    ref = slice(seq_txt + s["ref"].start, seq_txt + s["ref"].stop)
    src = slice(seq_txt + s["src"].start, seq_txt + s["src"].stop)
    return syn, ref, src


def _apply_orion_information_flow_mask(
    am: torch.Tensor,
    txt_j: slice,
    syn_j: slice,
    ref_j: slice,
    src_j: slice,
    has_src: bool,
    *,
    hard: bool,
    soft_beta: float,
) -> None:
    """In-place bias on joint attention logits [B,1,L,L]. Joint order: [text | image...].

    Matches paper grouping G1=text&syn, G2=source, G3=reference for allowed key groups:
    - txt & syn queries: no blocks (attend all keys).
    - src queries: block keys in txt_j and syn_j; may attend ref_j, src_j.
    - ref queries: block txt_j, syn_j, and src_j (if present); may attend ref_j only.
    """
    neg_inf = torch.finfo(am.dtype).min

    def block(q_slice: slice, k_slice: slice) -> None:
        if q_slice.start == q_slice.stop or k_slice.start == k_slice.stop:
            return
        if hard:
            am[:, :, q_slice, k_slice] = neg_inf
        else:
            am[:, :, q_slice, k_slice] = am[:, :, q_slice, k_slice] - soft_beta

    block(ref_j, txt_j)
    block(ref_j, syn_j)
    if has_src:
        block(ref_j, src_j)

    block(src_j, txt_j)
    block(src_j, syn_j)


class _BaseDoubleStreamProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss_cache = {}

    def _cache_losses(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self._loss_cache[k] = self._loss_cache.get(k, 0.0) + v

    def pop_losses(self) -> Dict[str, torch.Tensor]:
        out = self._loss_cache
        self._loss_cache = {}
        return out


class StageAProcessor(_BaseDoubleStreamProcessor):
    """Early layers: full Orion information-flow mask (text&syn | src | ref), see module comment."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("StageAProcessor requires PyTorch >=2.0")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        seg_slices_img: Optional[Dict[str, slice]] = None,
        soft_beta: Optional[float] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("StageAProcessor requires encoder_hidden_states")

        raw_seg = seg_slices_img or kwargs.get("seg_slices_img", None)
        seg = _coerce_seg_slices(raw_seg) if raw_seg else {}

        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        if attention_mask is not None:
            am = attention_mask.clone()
        else:
            am = None

        if seg:
            txt_j = slice(0, seq_txt)
            syn_j, ref_j, src_j = _joint_slices(seq_txt, seg)
            has_src = src_j.stop > src_j.start
            if am is None:
                B = hidden_states.shape[0]
                Ltot = joint_query.shape[1]
                am = torch.zeros(B, 1, Ltot, Ltot, device=hidden_states.device, dtype=hidden_states.dtype)
            _apply_orion_information_flow_mask(
                am, txt_j, syn_j, ref_j, src_j, has_src, hard=True, soft_beta=0.0
            )

        joint_hidden = _dispatch_joint_attention(
            joint_query, joint_key, joint_value, am, self._attention_backend, getattr(self, "_parallel_config", None)
        )
        joint_hidden = joint_hidden.flatten(2, 3).to(joint_query.dtype)

        txt_out = joint_hidden[:, :seq_txt, :]
        img_out = joint_hidden[:, seq_txt:, :]

        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)
        return img_out, txt_out


class StageBProcessor(StageAProcessor):
    """Mid layers: same hierarchy as Stage A; blocked pairs use soft_beta instead of -inf."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        seg_slices_img: Optional[Dict[str, slice]] = None,
        soft_beta: Optional[float] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        soft_beta = float(soft_beta if soft_beta is not None else kwargs.get("soft_beta", 6.0))
        if encoder_hidden_states is None:
            raise ValueError("StageBProcessor requires encoder_hidden_states")

        raw_seg = seg_slices_img or kwargs.get("seg_slices_img", None)
        seg = _coerce_seg_slices(raw_seg) if raw_seg else {}

        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        if attention_mask is not None:
            am = attention_mask.clone()
        else:
            am = None

        if seg:
            txt_j = slice(0, seq_txt)
            syn_j, ref_j, src_j = _joint_slices(seq_txt, seg)
            has_src = src_j.stop > src_j.start
            if am is None:
                B = hidden_states.shape[0]
                Ltot = joint_query.shape[1]
                am = torch.zeros(B, 1, Ltot, Ltot, device=hidden_states.device, dtype=hidden_states.dtype)
            _apply_orion_information_flow_mask(
                am, txt_j, syn_j, ref_j, src_j, has_src, hard=False, soft_beta=soft_beta
            )

        joint_hidden = _dispatch_joint_attention(
            joint_query, joint_key, joint_value, am, self._attention_backend, getattr(self, "_parallel_config", None)
        )
        joint_hidden = joint_hidden.flatten(2, 3).to(joint_query.dtype)

        txt_out = joint_hidden[:, :seq_txt, :]
        img_out = joint_hidden[:, seq_txt:, :]

        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)
        return img_out, txt_out


class TriOrthogonalLowRankProjector(nn.Module):
    """One QR on R^{d x 3r} yields orthonormal A1,A2,A3; trainable B1,B2,B3 for low-rank updates within each subspace."""

    def __init__(self, d: int, r: int = 256):
        super().__init__()
        A1, A2, A3 = self._init_orthogonal_As(d, r)
        self.A1 = nn.Parameter(A1, requires_grad=False)
        self.A2 = nn.Parameter(A2, requires_grad=False)
        self.A3 = nn.Parameter(A3, requires_grad=False)
        self.B1 = nn.Parameter(torch.zeros(r, r))
        self.B2 = nn.Parameter(torch.zeros(r, r))
        self.B3 = nn.Parameter(torch.zeros(r, r))
        self.alpha = 1.0

    @staticmethod
    def _init_orthogonal_As(d: int, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            M = torch.randn(d, 3 * r) / (d ** 0.5)
            Q, _ = torch.linalg.qr(M, mode="reduced")
            A1 = Q[:, :r].contiguous()
            A2 = Q[:, r : 2 * r].contiguous()
            A3 = Q[:, 2 * r : 3 * r].contiguous()
        return A1, A2, A3

    @staticmethod
    def _apply_orthogonal(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, alpha: float) -> torch.Tensor:
        delta = x @ A
        delta = delta @ B
        delta = delta @ A.T
        return x + alpha * delta


class OrionDecoupledDoubleStreamAttnProcessor2_0(_BaseDoubleStreamProcessor):
    _attention_backend = None
    _parallel_config = None

    def __init__(self, d: int, rank: int = 64):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("OrionDecoupledDoubleStreamAttnProcessor2_0 requires PyTorch >=2.0")
        self.proj = TriOrthogonalLowRankProjector(d, rank)
        self.synthesis_fusion_logits = nn.Parameter(torch.zeros(3))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        soft_beta: Optional[float] = None,
        seg_slices_img: Optional[Dict[str, slice]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("Decoupled processor requires encoder_hidden_states")

        raw_seg = seg_slices_img or kwargs.get("seg_slices_img", None)
        if raw_seg is None:
            base = OrionDoubleStreamAttnProcessor2_0()
            return base(attn, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, attention_mask, image_rotary_emb)

        seg = _coerce_seg_slices(raw_seg)
        seq_txt = encoder_hidden_states.shape[1]

        img_Q = attn.to_q(hidden_states)
        img_K = attn.to_k(hidden_states)
        img_V = attn.to_v(hidden_states)
        txt_Q = attn.add_q_proj(encoder_hidden_states)
        txt_K = attn.add_k_proj(encoder_hidden_states)
        txt_V = attn.add_v_proj(encoder_hidden_states)

        H = attn.heads
        img_Qh = img_Q.unflatten(-1, (H, -1))
        img_Kh = img_K.unflatten(-1, (H, -1))
        img_Vh = img_V.unflatten(-1, (H, -1))
        txt_Qh = txt_Q.unflatten(-1, (H, -1))
        txt_Kh = txt_K.unflatten(-1, (H, -1))
        txt_Vh = txt_V.unflatten(-1, (H, -1))

        if attn.norm_q is not None:
            img_Qh = attn.norm_q(img_Qh)
        if attn.norm_k is not None:
            img_Kh = attn.norm_k(img_Kh)
        if attn.norm_added_q is not None:
            txt_Qh = attn.norm_added_q(txt_Qh)
        if attn.norm_added_k is not None:
            txt_Kh = attn.norm_added_k(txt_Kh)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_Qh = apply_rotary_emb_qwen(img_Qh, img_freqs, use_real=False)
            img_Kh = apply_rotary_emb_qwen(img_Kh, img_freqs, use_real=False)
            txt_Qh = apply_rotary_emb_qwen(txt_Qh, txt_freqs, use_real=False)
            txt_Kh = apply_rotary_emb_qwen(txt_Kh, txt_freqs, use_real=False)

        img_Qm = img_Qh.flatten(2, 3)
        img_Km = img_Kh.flatten(2, 3)
        img_Vm = img_Vh.flatten(2, 3)
        txt_Km = txt_Kh.flatten(2, 3)
        txt_Vm = txt_Vh.flatten(2, 3)

        joint_query = torch.cat([txt_Qh, img_Qh], dim=1)
        joint_key = torch.cat([txt_Kh, img_Kh], dim=1)
        joint_value = torch.cat([txt_Vh, img_Vh], dim=1)
        joint_hidden = _dispatch_joint_attention(
            joint_query,
            joint_key,
            joint_value,
            attention_mask,
            self._attention_backend,
            getattr(self, "_parallel_config", None),
        )
        joint_hidden = joint_hidden.flatten(2, 3).to(joint_query.dtype)
        txt_out = joint_hidden[:, :seq_txt, :]
        img_out = joint_hidden[:, seq_txt:, :]

        s = seg
        syn = s["syn"]
        ref = s["ref"]
        src = s["src"]
        has_src = src.stop > src.start
        Lz = syn.stop - syn.start
        Oz_base = img_out[:, syn, :].clone()

        Q_syn = img_Qm[:, syn, :]
        K_syn = img_Km[:, syn, :]
        V_syn = img_Vm[:, syn, :]
        K_ref = img_Km[:, ref, :]
        V_ref = img_Vm[:, ref, :]
        Kt = txt_Km
        Vt = txt_Vm

        P = TriOrthogonalLowRankProjector
        pr = self.proj
        K_syn_p = P._apply_orthogonal(K_syn, pr.A1, pr.B1, pr.alpha)
        V_syn_p = P._apply_orthogonal(V_syn, pr.A1, pr.B1, pr.alpha)
        K_ref_p = P._apply_orthogonal(K_ref, pr.A2, pr.B2, pr.alpha)
        V_ref_p = P._apply_orthogonal(V_ref, pr.A2, pr.B2, pr.alpha)

        Q_ref_branch = P._apply_orthogonal(Q_syn, pr.A2, pr.B2, pr.alpha)
        if has_src:
            K_src = img_Km[:, src, :]
            V_src = img_Vm[:, src, :]
            K_src_p = P._apply_orthogonal(K_src, pr.A3, pr.B3, pr.alpha)
            V_src_p = P._apply_orthogonal(V_src, pr.A3, pr.B3, pr.alpha)
            Q_src_branch = P._apply_orthogonal(Q_syn, pr.A3, pr.B3, pr.alpha)
            joint_k = torch.cat([Kt, K_syn_p, K_ref_p, K_src_p], dim=1)
            joint_v = torch.cat([Vt, V_syn_p, V_ref_p, V_src_p], dim=1)
            joint_q = torch.cat([Q_ref_branch, Q_src_branch], dim=1)
        else:
            Q_alt = P._apply_orthogonal(Q_syn, pr.A1, pr.B1, pr.alpha)
            joint_k = torch.cat([Kt, K_syn_p, K_ref_p], dim=1)
            joint_v = torch.cat([Vt, V_syn_p, V_ref_p], dim=1)
            joint_q = torch.cat([Q_ref_branch, Q_alt], dim=1)

        joint_q = joint_q.unflatten(-1, (H, -1))
        joint_k = joint_k.unflatten(-1, (H, -1))
        joint_v = joint_v.unflatten(-1, (H, -1))

        joint_br = _dispatch_joint_attention(
            joint_q,
            joint_k,
            joint_v,
            None,
            self._attention_backend,
            getattr(self, "_parallel_config", None),
        )
        joint_br = joint_br.flatten(2, 3).to(joint_q.dtype)
        out1 = joint_br[:, :Lz, :]
        out2 = joint_br[:, Lz : 2 * Lz, :]

        w = torch.softmax(self.synthesis_fusion_logits, dim=0)
        Oz_blend = w[0] * Oz_base + w[1] * out1 + w[2] * out2
        img_out[:, syn, :] = Oz_blend

        img_out_proj = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out_proj = attn.to_out[1](img_out_proj)
        txt_out_proj = attn.to_add_out(txt_out)

        return img_out_proj, txt_out_proj


def set_orion_block_processors(model: "OrionEditTransformer2DModel", d: int, rank: int = 256, **kwargs):
    """Layers 0..STAGE_A-1: StageA; STAGE_A..STAGE_A+STAGE_B-1: StageB (soft_beta schedule); rest: decoupled."""
    kwargs.pop("use_gate", None)
    assert len(model.transformer_blocks) == 60, "This helper expects 60 layers."
    stage_b_end = STAGE_A_LAYERS + STAGE_B_LAYERS
    for idx, blk in enumerate(model.transformer_blocks):
        if idx < STAGE_A_LAYERS:
            blk.attn.processor = StageAProcessor()
        elif idx < stage_b_end:
            proc = StageBProcessor()
            proc.soft_beta = 8.0 - 6.0 * ((idx - STAGE_A_LAYERS) / max(STAGE_B_LAYERS - 1, 1))
            blk.attn.processor = proc
        else:
            blk.attn.processor = OrionDecoupledDoubleStreamAttnProcessor2_0(d=d, rank=rank)


# Backward compatibility for older call sites.
set_layered_processors_for_60_layers = set_orion_block_processors


def collect_decoupling_losses(model: OrionEditTransformer2DModel) -> Dict[str, torch.Tensor]:
    losses = {}
    for blk in model.transformer_blocks:
        proc = getattr(blk.attn, "processor", None)
        if hasattr(proc, "pop_losses") and callable(proc.pop_losses):
            ls = proc.pop_losses()
            for k, v in ls.items():
                losses[k] = losses.get(k, 0.0) + v
    return losses


def enable_decoupling_trainables(model: OrionEditTransformer2DModel):
    for blk in model.transformer_blocks:
        proc = getattr(blk.attn, "processor", None)
        if isinstance(proc, OrionDecoupledDoubleStreamAttnProcessor2_0):
            for n, p in proc.named_parameters(recurse=True):
                if n.startswith("proj.A"):
                    p.requires_grad_(False)
                elif n.startswith("proj.B") or n == "synthesis_fusion_logits":
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)


def get_decoupling_named_params(model: OrionEditTransformer2DModel) -> Dict[str, nn.Parameter]:
    out = {}
    for bi, blk in enumerate(model.transformer_blocks):
        proc = getattr(blk.attn, "processor", None)
        if isinstance(proc, OrionDecoupledDoubleStreamAttnProcessor2_0):
            prefix = f"transformer_blocks.{bi}.attn.processor."
            for n, p in proc.named_parameters(recurse=True):
                out[prefix + n] = p
    return out


