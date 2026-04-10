"""
Training defaults for OrionEdit Qwen pipeline.

`train.py` loads `TRAIN_DEFAULTS` and applies CLI overrides. Prefer editing this file
over hard-coding values in the training script.
"""

# --- paths & data (set exactly one of train_metadata_json / dataset_name / jsonl_for_train) ---
TRAIN_DEFAULTS = {
    "output_dir": "/data/zeyu/OrionEdit-main/output_0410",
    "pretrained_model_name_or_path": "/data/zeyu/models/Qwen-Image-Edit-2511",
    "train_metadata_json": "/data/zeyu/OrionEdit-main/dataset/metadata/metadata.json",
    "cache_dir": None,
    "dataset_name": None,
    "jsonl_for_train": None,
    "dataset_config_name": None,
    "image_column": "image",
    "conditioning_image_column": "conditioning_image",
    "caption_column": "text",
    "max_train_samples": None,
    # --- optimization ---
    "seed": 0,
    "resolution": 1024,
    "train_batch_size": 1,
    "num_train_epochs": 1,
    "max_train_steps": 3000,
    "checkpointing_steps": 200,
    "checkpoints_total_limit": None,
    "gradient_accumulation_steps": 6,
    "learning_rate": 7e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "lr_num_cycles": 1,
    "lr_power": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "scale_lr": False,
    "proportion_empty_prompts": 0.1,
    "guidance_scale": 1.0,
    "weighting_scheme": "logit_normal",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "rank": 256,
    "lora_layers": None,
    "use_lora_bias": False,
    "gaussian_init_lora": False,
    "gradient_checkpointing": True,
    "use_8bit_adam": True,
    "train_norm_layers": False,
    "allow_tf32": False,
    "mixed_precision": "bf16",
    "dataloader_num_workers": 4,
    "upcast_before_saving": False,
    "offload": True,
    "resume_from_checkpoint": "latest",
    # --- logging / hub ---
    "report_to": "wandb",
    "logging_dir": "logs",
    "tracker_project_name": "OrionEdit-qwen",
    "wandb_run_name": "OrionEdit-0410-1",
    "run_name": None,
    "log_dataset_samples": False,
    "push_to_hub": False,
    "hub_token": None,
    "hub_model_id": None,
    # --- periodic inference (disk + W&B); samples in INFERENCE_LOG_SAMPLES ---
    "inference_log_every": 200,
    "inference_num_steps": 30,
    # --- debug / legacy load ---
    "from_tuned_checkpoint": False,
}

WEIGHTING_SCHEME_CHOICES = ("sigma_sqrt", "logit_normal", "mode", "cosmap", "none")

# Samples for training-time inference logging (edit_prompt / reference_image / source_image).
# Paths can be absolute or relative to repo. Empty source_image => fusion.
INFERENCE_LOG_SAMPLES = [
    {
        "edit_prompt": "",
        "reference_image": "",
        "source_image": ""
    }
]
