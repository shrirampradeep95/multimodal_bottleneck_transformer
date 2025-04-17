# Multimodal Bottleneck Transformer (MBT)

**Official PyTorch implementation of**  
📄 _[Attention Bottlenecks for Multimodal Fusion (NeurIPS 2021)](https://arxiv.org/abs/2107.00135)_

---

## Overview

This repository implements the **Multimodal Bottleneck Transformer (MBT)** from scratch. MBT is a transformer-based architecture that uses **bottleneck tokens** to efficiently fuse visual and audio information. Key features include:

- Support for **AudioSet** and **VGGSound** datasets
- Use of **AST** for audio encoding and **ViT** for video encoding
- Bottleneck fusion blocks at each transformer layer
- Classification using CLS token fusion
- Fully modular, extensible, and dataset-agnostic

---

## 🔧 Architecture

The model processes **video frames** and **log-mel spectrograms** via separate transformer encoders. Bottleneck tokens mediate cross-modal attention:

      [Video Frames]     [Spectrograms]
            ↓                  ↓
       ViT Encoder         AST Encoder
            ↓                  ↓
    [RGB Tokens]         [Audio Tokens]
            ↘              ↙
    ↘   Bottleneck Tokens   ↙
    →→→→→→ Fusion Layers →→→→→→
                 ↓
       [CLS Tokens from both]
                 ↓
         Linear Classifier


---

## Core Concepts

### Bottleneck Attention
Instead of full cross-attention (quadratic cost), a small number of shared **bottleneck tokens** attend to both modalities to transfer relevant information.

### CLS Token Fusion
Final classification is done using a linear head on top of the **averaged logits** from audio and video CLS tokens.

### 🔍 Positional Encoding & Patchification
- **Video**: 8 frames at 25 FPS → 14×14 patches → 1568 tokens
- **Audio**: t-second log-mel spectrogram → 400 tokens (16×16 patches)

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt

**Key Packages**
torch — PyTorch deep learning framework
timm — Pretrained Vision Transformers (used for ViT)
einops — Tensor manipulation library (used for rearranging tokens)
torchaudio — For audio I/O and spectrogram preprocessing
opencv-python — For video frame extraction and processing
scikit-learn — Evaluation metrics like average precision
tqdm — Progress bar for training and data loading
numpy, pandas — Standard scientific computing stack

## **Loss Functions**

BCEWithLogitsLoss for AudioSet (multi-label classification)
CrossEntropyLoss for VGGSound (single-label classification)

