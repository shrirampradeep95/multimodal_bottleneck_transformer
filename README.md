# Multimodal Bottleneck Transformer (MBT)

A PyTorch-based implementation of the **Multimodal Bottleneck Transformer** for audio-visual classification. This project supports training and evaluating models on datasets like **AudioSet** and **VGGSound**, using audio, video, or fused features with cross-modal bottleneck fusion.

## 🧠 Key Features

- Multimodal fusion with bottleneck tokens (MBT)
- Supports AudioSet (multi-label) and VGGSound (single-label)
- Training options for audio-only, video-only, or both
- Plug-and-play architecture using PyTorch modules
- Visualization of training loss and evaluation metrics
---

## 📂 Project Structure

multimodal_bottleneck_transformer/ │ ├── utils/ │ ├── preprocessing/ │ │ ├── audioset_data_preprocessing.py │ │ ├── vgg_sound_data_preprocessing.py │ │ └── video_data_preprocessing.py │ └── train_eval/ │ ├── trainer_evaluator.py │ └── generate_plots.py │ ├── utils/model/ │ ├── audio_model.py │ ├── video_model.py │ ├── bottleneck_fusion.py │ └── multimodal_transformer.py │ ├── runner.py (or mbt_runner.py) # Entry point └── README.md
