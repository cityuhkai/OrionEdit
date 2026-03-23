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



## 📖 Introduction

We present OrionEdit, a unified framework for cross-image editing that combines symmetric orthogonal subspace disentanglement with reverse-causal attention, where information-flow masks enforce unidirectional dependencies in the latent space.

OrionEdit is deployed on standard diffusion backbones and supports zero-shot multi-reference editing, while outperforming open-source baselines in fidelity and compositional consistency.


## 🚀 Quick Start

Requirements and Installation
First, install the necessary dependencies:


## 🗂️ OrionEditBench
## 🗂️ OrionEditBench

To support research on cross-image editing, we construct a dataset based on **reference–source–synthesis triplets**. 
Given the lack of large-scale datasets in this format, our data is built from a combination of partially public sources 
(e.g., Subjects200K, ShareGPT-4o-Image, OmniContext, DeepFashion) and curated synthetic pairs generated using Nano-banana and GPT-4o.

The released dataset covers diverse editing scenarios, with a particular emphasis on **visual attribute transfer**, 
as well as fusion-based generation and style alignment tasks.

We release a **subset of the OrionEdit dataset** to facilitate training, ablation studies, and qualitative evaluation.  
The full benchmark (including evaluation splits and protocols) is not publicly available to ensure fair and consistent comparison.

The dataset is hosted on Hugging Face:
👉 https://huggingface.co/datasets


## 🤗 Disclaimer

This repository is built upon [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and is released under the Apache 2.0 License, we thank [Magiclight.AI](https://magiclight.ai) for their support.

This project is intended for academic research and the AIGC community. Most images are AI-generated or from public datasets. For any concerns, please contact us; we will promptly review and remove inappropriate content.








## ⭐ Citation

If OrionEdit inspires your research 🤔, please consider giving this repo a ⭐ and citing our work:

```bibtex
@article{
> Our paper has been accepted to CVPR 2026 main track. 
The official citation will be released upon publication.
}