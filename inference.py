"""
Batch inference aligned with ``train.run_orion_inference_log`` (same pipeline kwargs).

Loads LoRA + decoupled processor weights from a training checkpoint (e.g. ``checkpoints/step-400``),
runs all samples from a JSON list (same schema as training inference_log: ``edit_prompt``, ``reference_image``, ``source_image``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image

from models._pipeline_qwen_edit_2509 import QwenImageEditPlusPipeline
from models.dns_transformer_qwen_2509 import (
    dns_QwenImageTransformer2DModel_2509,
    set_layered_processors_for_60_layers,
)

try:
    from safetensors.torch import load_file as load_sft
except Exception:
    load_sft = None

_REPO = Path(__file__).resolve().parent
_DEFAULT_PRETRAINED = "/data/zeyu/models/Qwen-Image-Edit-2511"
_DEFAULT_CHECKPOINT = _REPO / "output_0406" / "checkpoints" / "step-400"
_DEFAULT_SAMPLES_JSON = _REPO / "-tmp_infer.json"


def load_processor_weights_from_safetensors(transformer: torch.nn.Module, sft_path: str) -> None:
    if load_sft is None:
        print("[WARN] safetensors not available; skip processor loading.")
        return
    if not os.path.isfile(sft_path):
        print(f"[INFO] no processor safetensors at: {sft_path} (skip)")
        return
    state = load_sft(sft_path, device="cpu")
    dec_sd = {}
    for k, v in state.items():
        if k.startswith("transformer.") and "attn.processor" in k:
            new_k = k.replace("transformer.", "", 1)
            dec_sd[new_k] = v
    if not dec_sd:
        print("[INFO] no 'attn.processor' keys in processor safetensors.")
        return
    missing, unexpected = transformer.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        print(f"[INFO] processor load: missing={len(missing)}, unexpected={len(unexpected)}")


def _step_from_checkpoint_dir(ckpt_dir: Path) -> int:
    name = ckpt_dir.name
    m = re.match(r"^step-(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^step-final-(\d+)$", name)
    if m:
        return int(m.group(1))
    return 0


def _resolve_image_path(p: str, json_dir: Path) -> Path:
    path = Path(p)
    if path.is_file():
        return path
    cand = json_dir / p
    if cand.is_file():
        return cand
    return path


def load_samples_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def build_pipeline(
    pretrained: str,
    checkpoint_dir: Path,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> QwenImageEditPlusPipeline:
    qwen_transformer = dns_QwenImageTransformer2DModel_2509.from_pretrained(
        pretrained,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    set_layered_processors_for_60_layers(qwen_transformer, d=3072, rank=256, use_gate=False)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        pretrained,
        transformer=qwen_transformer,
        torch_dtype=weight_dtype,
    )

    lora_path = checkpoint_dir / "pytorch_lora_weights.safetensors"
    if not lora_path.is_file():
        raise FileNotFoundError(f"Missing LoRA weights: {lora_path}")
    pipeline.load_lora_weights(str(lora_path), torch_dtype=weight_dtype)

    proc_path = checkpoint_dir / "pytorch_processor_weights.safetensors"
    load_processor_weights_from_safetensors(pipeline.transformer, str(proc_path))

    _excl = list(getattr(pipeline, "_exclude_from_cpu_offload", None) or [])
    if "transformer" not in _excl:
        pipeline._exclude_from_cpu_offload = _excl + ["transformer"]
    pipeline.enable_sequential_cpu_offload(device=device)
    pipeline.set_progress_bar_config(disable=None)
    return pipeline


def run_sample_like_train(
    pipeline: QwenImageEditPlusPipeline,
    sample: dict,
    json_dir: Path,
    device: torch.device,
    weight_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    """Same inputs as ``train.run_orion_inference_log`` (no width/height/generator)."""
    prompt = sample.get("edit_prompt") or sample.get("prompt", "")
    ref = sample["reference_image"]
    src = sample.get("source_image", "") or ""

    if isinstance(ref, list):
        ref_in = [Image.open(_resolve_image_path(str(p), json_dir)).convert("RGB") for p in ref]
    else:
        ref_path = _resolve_image_path(str(ref), json_dir)
        ref_in = Image.open(ref_path).convert("RGB")

    src_stripped = str(src).strip()
    if src_stripped:
        src_path = _resolve_image_path(src_stripped, json_dir)
        src_in = Image.open(src_path).convert("RGB")
    else:
        src_in = None

    with torch.autocast(device_type=device.type, dtype=weight_dtype, enabled=True):
        out = pipeline(
            prompt=prompt,
            reference_image=ref_in,
            source_image=src_in,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=4.0,
            negative_prompt=" ",
            guidance_scale=guidance_scale,
        )
    return out.images[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OrionEdit batch inference (train.py inference_log parity).")
    p.add_argument("--pretrained-model", type=str, default=_DEFAULT_PRETRAINED)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Training checkpoint dir containing pytorch_lora_weights.safetensors",
    )
    p.add_argument("--samples-json", type=Path, default=_DEFAULT_SAMPLES_JSON)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, save under this dir / inference_log / step_<n>/. "
        "Default: <checkpoint_dir>/../inference_log/step_<n>/",
    )
    p.add_argument("--inference-num-steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = args.checkpoint_dir.resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint-dir not found: {ckpt_dir}")

    samples_path = args.samples_json.resolve()
    if not samples_path.is_file():
        raise FileNotFoundError(f"samples-json not found: {samples_path}")

    step_tag = _step_from_checkpoint_dir(ckpt_dir)
    if args.output_dir is not None:
        log_dir = args.output_dir.resolve() / "inference_log" / f"step_{step_tag}"
    else:
        out_root = ckpt_dir.parent.parent
        log_dir = out_root / "inference_log" / f"step_{step_tag}"

    json_dir = samples_path.parent
    samples = load_samples_json(samples_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Checkpoint: {ckpt_dir}")
    print(f"Samples:    {samples_path} ({len(samples)} items)")
    print(f"Output:     {log_dir}")

    pipeline = build_pipeline(
        args.pretrained_model,
        ckpt_dir,
        weight_dtype,
        device,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        try:
            pil_img = run_sample_like_train(
                pipeline,
                sample,
                json_dir=json_dir,
                device=device,
                weight_dtype=weight_dtype,
                num_inference_steps=args.inference_num_steps,
                guidance_scale=args.guidance_scale,
            )
            out_path = log_dir / f"sample_{i:02d}.png"
            pil_img.save(out_path)
            print(f"saved {out_path}")
        except Exception as e:
            print(f"[WARN] sample {i} failed: {e}")

    try:
        pipeline.remove_all_hooks()
    except Exception:
        pass
    print(f"Done. Inference log layout matches train.py under {log_dir}")


if __name__ == "__main__":
    main()
