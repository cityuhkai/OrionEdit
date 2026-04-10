import os

import torch
from PIL import Image

from models.pipeline_orion_edit import OrionEditPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

pipeline = OrionEditPipeline.from_orion_pretrained(
    base_model="Qwen/Qwen-Image-Edit-2511",
    orion_repo="ZeyuJiang1/OrionEdit-qwen",
    torch_dtype=dtype,
    device=device,
)

source_image = "/path/to/image_to_be_edited"
# Supports multiple references, best for 2. 
reference_image = [
    "/path/to/reference_image_1",
    "/path/to/reference_image_2",
]
# fusion task:
# reference_image = ["/path/to/reference_image_1", "/path/to/reference_image_2"]
# keep source_image = "" for fusion task

reference_image = (
    [Image.open(p).convert("RGB") for p in reference_image]
    if isinstance(reference_image, list)
    else Image.open(reference_image).convert("RGB")
)
source_image = Image.open(source_image).convert("RGB") if str(source_image).strip() else None

with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
    image = pipeline(
        prompt="A cinematic illustration of a fox explorer discovering a glowing ancient observatory.",
        reference_image=reference_image,
        source_image=source_image,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        negative_prompt=" ",
        guidance_scale=1.0,
    ).images[0]

image.save("output_image_orion_edit.png")
print("image saved at", os.path.abspath("output_image_orion_edit.png"))
