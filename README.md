🚀 Multimodal Bottleneck Transformer (MBT)
Official PyTorch implementation of the paper
📄 Attention Bottlenecks for Multimodal Fusion (NeurIPS 2021)

🎯 Overview
This repository implements the Multimodal Bottleneck Transformer (MBT) from scratch, a transformer-based architecture that efficiently fuses audio and visual modalities using bottleneck tokens. This implementation supports:

AudioSet and VGGSound datasets

AST for audio and ViT for video

Bottleneck fusion across transformer layers

CLS token-based classification

Fully modular and extensible design

🔧 Architecture Overview
The model is designed to process video frames and log-mel spectrograms using modality-specific encoders, followed by fusion through bottleneck attention:

css
Copy
Edit
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
Fusion happens via cross-attention to bottleneck tokens, which mediate the interaction between modalities while maintaining computational efficiency.

📁 Project Structure
bash
Copy
Edit
multimodal_bottleneck_transformer/
├── utils/
│   ├── model/
│   │   ├── audio_model.py        # AST encoder
│   │   ├── video_model.py        # ViT encoder
│   │   └── bottleneck_fusion.py  # Bottleneck fusion blocks
│   ├── preprocessing/
│   │   └── video_data_preprocessing.py  # Audio-video loader
│   └── train_eval/
│       └── trainer_evaluator.py  # Training + evaluation logic
├── mbt_runner_class.py           # Main class for training
├── README.md                     # You're here
🧠 Key Concepts
Bottleneck Attention
MBT reduces full attention complexity by introducing a small number of fusion tokens that attend to both modalities and relay relevant information.

CLS Token Fusion
We average the logits from the CLS token of both encoders for final classification.

Positional Encoding & Patchification
Audio: 128×100t spectrogram → 16×16 patches → 400 tokens

Video: 8 frames → 14×14 patches each → 1568 tokens

📦 Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Core packages:

torch, timm

einops

torchaudio, opencv

scikit-learn, tqdm

📊 Datasets Supported

Dataset	Frames	Audio Length (t)	Tokens
AudioSet	8	t seconds	400 spectrogram tokens
VGGSound	8	t seconds	1568 image patch tokens
Spectrogram: 16kHz mono audio → log-mel with 25ms window, 10ms hop

RGB frames: Sampled at 25 FPS, resized to 224×224

🏋️‍♂️ Training
bash
Copy
Edit
python mbt_runner_class.py
Configure hyperparameters and dataset settings in your parameters dictionary or config file.

✅ Evaluation
AudioSet: Multi-label → BCEWithLogitsLoss, mAP

VGGSound: Single-label → CrossEntropyLoss, Top-1 Accuracy
