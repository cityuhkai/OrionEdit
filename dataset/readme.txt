Description of the dataset directory structure:

The `dataset` directory is used to store data for training and evaluation. The typical files and subdirectories are as follows:

- metadata/metadata.json  
  This file stores metadata for the dataset. Each line (or entry) is a JSON object representing a data sample, with the following main fields:
    - t2i_prompt: Legacy field for original text-to-image prompt.
    - edit_prompt: Textual description for editing tasks.
    - reference_image: Path or list of paths to reference images (string or list of strings).
    - source_image: Optional, path to source image (empty or absent indicates a fusion task).
    - output_image: Path to the target image for training.
    - width/height: Image size (legacy fields).
    - img_ids: Image ID (legacy field).

- images/
  Contains image files (e.g., .jpg, .png). The `reference_image`, `source_image`, and `output_image` fields in metadata.json refer to the relative paths of these images within the `images/` directory.

Example structure:

dataset/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── metadata/
    └── metadata.json

You may extend the directory with custom data as needed, but make sure the fields and paths in `metadata.json` are consistent with the actual data.