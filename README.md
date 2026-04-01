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
    <a href="https://huggingface.co/">
        <img alt="Build" src="https://img.shields.io/badge/🤗-HF%20Model-yellow">
    </a>    
</p>

![The teaser figure of OrionEdit.](assets/teaser.png)


## 🔥 News

- **2026.3.22**: The repo has been released!
- **2026.4.01**: We release a subset of the OrionEditBench metadata, including the AI-generated data for attribute transfer task!



## 📖 Introduction

We present OrionEdit, a unified framework for cross-image editing that combines symmetric orthogonal subspace disentanglement with reverse-causal attention, where information-flow masks enforce unidirectional dependencies in the latent space.

OrionEdit is deployed on standard diffusion backbones and supports zero-shot multi-reference editing, while outperforming open-source baselines in fidelity and compositional consistency.




## 🚀 Quick Start

Requirements and Installation
First, install the necessary dependencies:


## Inference



## 🗂️ OrionEditBench

We construct a dataset based on **reference–source–synthesis triplets** to support cross-image editing. Due to the lack of large-scale data in this format, our collection combines samples adapted from existing public datasets (e.g., Subjects200K, ShareGPT-4o-Image, OmniContext, DeepFashion) with a substantial portion of curated synthetic pairs generated using Nano-banana and GPT-4o.

The released subset covers diverse editing scenarios, with a focus on **visual attribute transfer**, along with fusion-based generation and style alignment.
We release part of the dataset for training and analysis, the dataset is hosted on Hugging Face, click 👉 [here](https://huggingface.co/datasets/ZeyuJiang1/OrionEditBench). 

To reduce training overhead, some of multiple reference images are pre-composed into a single input (optionally with background removal) so they share a unified branch; examples are shown below (from left to right: reference image(s), source image, and synthesized result).

![The example of dataset.](assets/dataset.png)



## 🤗 Disclaimer

This repository is built upon [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and is released under the Apache 2.0 License. We thank [Magiclight.AI](https://magiclight.ai) for their support in dataset collection and training resources. This project is intended for academic research and the broader AIGC community. Most of the released images are AI-generated or sourced from public datasets. For any concerns, please contact us; we will promptly review and remove inappropriate content.




## ⭐ Citation

If OrionEdit inspires your research 🤔, please consider giving this repo a ⭐ and citing our work:

```bibtex
@article{
> Our paper has been accepted to CVPR 2026 main track. 
The official citation will be released upon publication.
}