# Modified from the training script provided by diffusers FLUX.1 dev.

import json
import random

from torch.utils.data import Dataset


def _is_blank_source(src) -> bool:
    if src is None:
        return True
    if isinstance(src, str) and not src.strip():
        return True
    return False


class CustomDataset(Dataset):
    """JSON rows aligned with ``dataset/metadata/metadata.json``.

    * ``t2i_prompt`` — legacy field, original text-to-image prompt.
    * ``edit_prompt`` — edit prompt for the text encoder.
    * ``reference_image`` — str or list of str paths (fusion uses list).
    * ``source_image`` — optional; empty / missing ⇒ fusion; non-empty ⇒ edit (exactly one reference).
    * ``output_image`` — synthesis target for training.
    * ``width`` — legacy field, image width.
    * ``height`` — legacy field, image height.
    * ``img_ids`` — legacy field, image IDs.

    """

    def __init__(self, data_meta_paths=None, shuffle_seed=0):
        super().__init__()
        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r") as f:
                vid_meta.extend(json.load(f))

        rng = random.Random(int(shuffle_seed))
        rng.shuffle(vid_meta)
        self.vid_meta = vid_meta

    def __getitem__(self, index):
        m = self.vid_meta[index]
        if "edit_prompt" not in m:
            raise KeyError(
                f"Sample at index {index} must contain 'edit_prompt' (and reference_image / output_image). "
                f"Legacy cond_1/cond_2 format is no longer supported."
            )

        prompt = m["edit_prompt"]
        ref = m["reference_image"]
        out = m["output_image"]
        src = m.get("source_image", "")

        if isinstance(ref, list):
            ref_list = [str(x).strip() for x in ref if x is not None and str(x).strip()]
        elif ref is not None and str(ref).strip():
            ref_list = [str(ref).strip()]
        else:
            ref_list = []

        if _is_blank_source(src):
            source_image = ""
            if len(ref_list) < 1:
                raise ValueError(
                    f"Fusion sample (empty source_image) needs at least one reference_image (index {index})."
                )
        else:
            source_image = str(src).strip()
            if len(ref_list) != 1:
                raise ValueError(
                    f"Edit sample needs exactly one reference_image when source_image is set (index {index}); "
                    f"got {len(ref_list)}."
                )

        return {
            "prompt": prompt,
            "reference_image": ref_list,
            "source_image": source_image,
            "image": out,
        }

    def __len__(self):
        return len(self.vid_meta)
