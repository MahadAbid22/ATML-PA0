# ATML Assignment 0: Empirical Investigation of CNNs, Vision Transformers, Contrastive Learning, and VAEs

**Course:** Advanced Topics in Machine Learning  
**Author:** Muhammad Mahad Abid  
**Institution:** Department of Computer Science, Lahore University of Management Sciences (LUMS)  
**Email:** 28100303@lums.edu.pk  
**Repository:** [MahadAbid22/ATML-PA0](https://github.com/MahadAbid22/ATML-PA0)

---

## Overview

This repository contains the implementation, experimental notebooks, and visual artifacts for **ATML Assignment 0**. The codebase empirically investigates how modern deep learning architectures learn, structure, and manipulate visual representations across four core paradigms:

1. **Residual Architectures (ResNet-152):** Feature transferability on CIFAR-10, gradient explosion dynamics when residual connections are severed in FP32 precision, hierarchical representation disentanglement across network depth via t-SNE, and catastrophic forgetting under full backbone fine-tuning.
2. **Vision Transformers (ViT-Base/16):** Patch tokenization, multi-head self-attention (MHSA) interpretability across individual attention heads, structural robustness under random, center, and background patch masking, and linear probe comparisons between `[CLS]` token representations and mean-pooled spatial patch tokens.
3. **Multimodal Contrastive Learning (CLIP ViT-B/32):** Zero-shot transfer across prompt engineering templates on STL-10, geometric characterization of the cross-modal "modality gap" in 512D unit hypersphere space, Orthogonal Procrustes alignment via SVD rotation, and class-level confusion matrix analysis.
4. **Deep Generative Modeling (Variational Autoencoders):** A pure PyTorch implementation of an MLP-based VAE on MNIST, reparameterization trick, negative ELBO optimization (Bernoulli BCE reconstruction loss + Gaussian KL divergence), 2D latent space topology, reconstruction fidelity, and random prior sample generation.

---

## Repository Structure

```text
ATML-PA0/
├── data/                      # Auto-downloaded benchmark datasets (CIFAR-10, STL-10, MNIST)
├── notebooks/                 # Standalone Jupyter notebooks for each task
│   ├── task1_resnet.ipynb     # Task 1: ResNet-152 experiments
│   ├── task2_vit.ipynb        # Task 2: Vision Transformer experiments
│   ├── task3_clip.ipynb       # Task 3: CLIP multimodal zero-shot experiments
│   └── task4_vae.ipynb        # Task 4: Variational Autoencoder experiments
├── report/
│   └── figures/               # Generated experiment plots and visualizations
├── requirements.txt           # Python environment dependencies
└── README.md                  # Setup and reproduction instructions
```

---

## Environment Setup

### 1. Prerequisites
- Python **3.10+**
- NVIDIA GPU with CUDA support recommended (mixed precision AMP is enabled where applicable to optimize memory and runtime).

### 2. Installation
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/MahadAbid22/ATML-PA0.git
cd ATML-PA0

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install official OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

---

## Notebooks & Execution Guide

All experiments are organized into standalone notebooks inside [`notebooks/`](notebooks/). Benchmark datasets (CIFAR-10, STL-10, MNIST) download automatically into [`data/`](data/) during execution.

```bash
jupyter lab
```

### [Task 1: Inner Workings of ResNet-152](notebooks/task1_resnet.ipynb)
- **Notebook:** `notebooks/task1_resnet.ipynb`
- **Dataset:** CIFAR-10
- **Experiments:**
  1. **Transfer Baseline:** Evaluates frozen ImageNet-pretrained ResNet-152 convolutional layers with a newly initialized linear classification head.
  2. **Gradient Flow Ablation:** Dynamically monkey-patches all 50 `Bottleneck` residual blocks to sever the identity shortcut ($y = \mathcal{F}(x)$). Tracks $L_2$ gradient norms of `conv1` vs. `fc` across 5 training batches in pure **FP32** precision to demonstrate severe gradient explosion.
  3. **Layer-wise t-SNE:** Uses PyTorch forward hooks on `layer1`, `layer3`, and `layer4` to extract feature maps, pool them, and visualize class clustering across depth.
  4. **Fine-Tuning Regimes:** Compares head-only tuning against full backbone fine-tuning (highlighting catastrophic forgetting on small datasets) and training from scratch.

### [Task 2: Understanding Vision Transformers (ViT)](notebooks/task2_vit.ipynb)
- **Notebook:** `notebooks/task2_vit.ipynb`
- **Model & Dataset:** `google/vit-base-patch16-224`, CIFAR-10
- **Experiments:**
  1. **Zero-Shot Inference:** Splits input images into $16 \times 16$ non-overlapping patches, appends the learnable `[CLS]` token, and performs fine-grained classification.
  2. **Attention Extraction & Saliency:** Extracts Layer 11 attention weights for the `[CLS]` token. Generates both the 12-head averaged heatmap and individual head heatmaps (Heads 0–3) to demonstrate semantic and contextual specialization.
  3. **Patch Masking Perturbations:** Masks 25% (49 patches) under three conditions: Random, Center ($7 \times 7$), and Background, comparing classification robustness and failure modes.
  4. **Linear Probing (`[CLS]` vs. Mean Pooling):** Extracts representations from the frozen ViT and trains linear probes to compare `[CLS]` token accuracy against global average patch pooling on CIFAR-10.

### [Task 3: Multimodal Representation Learning with CLIP](notebooks/task3_clip.ipynb)
- **Notebook:** `notebooks/task3_clip.ipynb`
- **Model & Dataset:** OpenAI `ViT-B/32` CLIP, STL-10 (full 8,000-image test set)
- **Experiments:**
  1. **Prompt Engineering:** Tests zero-shot classification across 4 prompt formulations: plain label, `"a photo of a {}."`, descriptive low-resolution prompt, and a 6-template ensemble.
  2. **Modality Gap Characterization:** Computes normalized 512D image and text embeddings for 100 test pairs, measures centroid Euclidean distance, and visualizes separation via t-SNE.
  3. **Orthogonal Procrustes Alignment:** Solves $R^* = \arg\min_R \|X_{\text{img}} R - X_{\text{txt}}\|_F^2$ via SVD to rotate image features into the text subspace, evaluating whether closing the gap improves downstream zero-shot accuracy.
  4. **Error Analysis:** Generates a normalized confusion matrix to identify semantic confusion patterns (e.g. within animal and vehicle super-categories).

### [Task 4: Deep Generative Modeling with VAEs](notebooks/task4_vae.ipynb)
- **Notebook:** `notebooks/task4_vae.ipynb`
- **Dataset:** MNIST
- **Experiments:**
  1. **From-Scratch Architecture:** Implements a symmetric MLP VAE ($784 \to 400 \to 2 \to 400 \to 784$) with the reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
  2. **ELBO Optimization:** Trains for 15 epochs minimizing Bernoulli Binary Cross-Entropy reconstruction loss plus closed-form Gaussian KL divergence.
  3. **2D Latent Space Visualization:** Directly projects test embeddings $\mu(x)$ onto a 2D scatter plot color-coded by digit class.
  4. **Reconstructions & Sample Synthesis:** Evaluates reconstruction fidelity on unseen test digits (illustrating the 4 $\to$ 9 failure mode under an extreme 2D bottleneck) and decodes random vectors sampled from prior $\mathcal{N}(0, I)$.

---

## Experiment Visualizations & Generated Artifacts

Executing the notebooks automatically saves high-resolution experiment figures into [`report/figures/`](report/figures/). The table below details each generated artifact and its source:

| Saved Visualization | Generating Notebook & Cell | Description & Empirical Finding |
|:---|:---|:---|
| [`task1_exploding_gradients.png`](report/figures/task1_exploding_gradients.png) | `task1_resnet.ipynb` (Cell 5) | FP32 gradient norms showing gradient explosion at `conv1` (norm > 12,800) when Bottleneck skip connections are severed |
| [`task1_tsne_features.png`](report/figures/task1_tsne_features.png) | `task1_resnet.ipynb` (Cell 6) | 2D t-SNE projections demonstrating layer-wise feature progression from entangled low-level features (`layer1`) to class-separable clusters (`layer4`) |
| [`task2_averaged_map.png`](report/figures/task2_averaged_map.png) | `task2_vit.ipynb` (Cell 1) | ViT Layer 11 `[CLS]` self-attention heatmap averaged across all 12 heads showing diffuse foreground and background attention |
| [`task2_attention_heads.png`](report/figures/task2_attention_heads.png) | `task2_vit.ipynb` (Cell 1) | Attention heatmaps for Heads 0–3 showing functional specialization (semantic localization vs. fine anatomy vs. broad context) |
| [`task2_patch_masking.png`](report/figures/task2_patch_masking.png) | `task2_vit.ipynb` (Cell 2) | ViT classification and confidence under Random, Center ($7 \times 7$), and Background patch masking |
| [`task2_pooling_comparison.png`](report/figures/task2_pooling_comparison.png) | `task2_vit.ipynb` (Cell 3) | Linear probe classification on CIFAR-10: Mean-pooled spatial patch tokens (97.00%) outperforming `[CLS]` token (95.95%) |
| [`task3_zero_shot.png`](report/figures/task3_zero_shot.png) | `task3_clip.ipynb` (Cell 1) | STL-10 zero-shot accuracy across prompt strategies; standard `"a photo of a {}."` achieves best performance (97.31%) |
| [`task3_modality_gap.png`](report/figures/task3_modality_gap.png) | `task3_clip.ipynb` (Cell 2) | t-SNE of image-text embeddings before and after Orthogonal Procrustes alignment (centroid gap reduced by 86%) |
| [`task3_confusion_matrix.png`](report/figures/task3_confusion_matrix.png) | `task3_clip.ipynb` (Cell 3) | Normalized confusion matrix on STL-10 showing error concentration within semantic super-categories |
| [`task4_latent_space.png`](report/figures/task4_latent_space.png) | `task4_vae.ipynb` (Cell 1) | 2D latent space manifold on MNIST showing isolated clusters for digits 1 and 7 and overlapping boundaries for 4/9 and 3/5/8 |
| [`task4_reconstructions.png`](report/figures/task4_reconstructions.png) | `task4_vae.ipynb` (Cell 1) | Original vs. reconstructed MNIST digits showing overall structure retention alongside blurriness and 4 $\to$ 9 loop closure |
| [`task4_generations.png`](report/figures/task4_generations.png) | `task4_vae.ipynb` (Cell 1) | Decoded MNIST digits sampled from standard normal prior $\mathcal{N}(0, I)$ demonstrating continuous manifold transitions |

---

## Summary of Empirical Results

| Task / Experiment | Configuration / Model | Metric | Value |
|:---|:---|:---|:---:|
| **Task 1: Baseline Transfer** | ResNet-152 (Frozen Conv, Head-Only) | CIFAR-10 Val Accuracy | **84.19%** |
| **Task 1: Full Fine-Tuning** | ResNet-152 (Unfrozen Backbone) | CIFAR-10 Val Accuracy | **81.63%** |
| **Task 1: Gradient Ablation** | ResNet-152 (Severed Skips, Batch 1) | `conv1` / `fc` Grad Norm Ratio | **13,222.8x** (Exploding) |
| **Task 2: Feature Pooling** | ViT-Base/16 (`[CLS]` Token) | CIFAR-10 Linear Probe Acc | **95.95%** |
| **Task 2: Feature Pooling** | ViT-Base/16 (Mean-Pooled Patches) | CIFAR-10 Linear Probe Acc | **97.00%** |
| **Task 3: Prompt Engineering** | CLIP ViT-B/32 (`"a photo of a {}."`) | STL-10 Zero-Shot Acc (Full) | **97.31%** |
| **Task 3: Prompt Engineering** | CLIP ViT-B/32 (Baseline `"{}"`) | STL-10 Zero-Shot Acc (Full) | **96.26%** |
| **Task 3: Modality Gap** | CLIP ViT-B/32 (Raw Centroid Distance) | Euclidean Distance | **1.0449** |
| **Task 3: Modality Gap** | CLIP ViT-B/32 (Procrustes Aligned) | Euclidean Distance | **0.1449** (-86.1%) |
| **Task 3: Aligned Inference** | CLIP ViT-B/32 (Rotated Features $z_{\text{img}} R$) | STL-10 Zero-Shot Acc (Full) | **96.74%** |
| **Task 4: Generative VAE** | MLP VAE ($d=2$, 15 epochs) | Average Loss (Negative ELBO) | **~146.4** |

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.
3. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*, 8748–8763.
4. Kingma, D. P., & Welling, M. (2019). An introduction to variational autoencoders. *Foundations and Trends in Machine Learning*, 12(4), 307–392.
5. Doersch, C. (2016). Tutorial on variational autoencoders. *arXiv:1606.05908*.
