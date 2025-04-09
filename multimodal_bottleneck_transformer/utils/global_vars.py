# Import relevant libraries
import torch

# Complete configuration dictionary for MBT runner and model
parameters = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dataset": "AudioSet",  # Dataset to use: "AudioSet" or "vggsound"
    "modality": "av",  # Modality type: "av" for audiovisual, "a" for audio-only, "v" for video-only
    "fps": 25,  # FPS and duration control for frame sampling and audio slicing
    "t_seconds": 1,  # FPS and duration control for frame sampling and audio slicing
    "num_frames": 8,  # FPS and duration control for frame sampling and audio slicing
    "num_bottlenecks": 4,  # Number of bottleneck tokens used in fusion transformer
    "transformer_layers": 12,  # Number of transformer layers in the model
    "vgg_sound_lr": 0.01,  # Learning rates for specific dataset types
    "audio_set_lr": 1e-4,  # Learning rates for specific dataset types

    # Training control
    "batch_size": 1,
    "epochs": 50,
    "num_workers": 4,
    "videos_to_use": None,

    # AudioSet-specific paths
    "audio_set_paths": {
        "train_video_path": "data/audioset/balanced_train_segments",
        "test_video_path": "data/audioset/eval_segments",
        "train_label_path": "data/audioset/balanced_train_segments.csv",
        "test_label_path": "data/audioset/eval_segments.csv",
        "class_label_mapping": "data/audioset/class_labels_indices.csv"
    },

    # VGGSound-specific paths
    "vgg_sound_paths": {
        "video_dir": "data/vggsound/vggsound_videos",
        "label_file": "data/vggsound/vggsound.csv",
    },

    # Epic Kitchens dataset
    "epic_kitchens_paths": {
        "train_video_path": "data/epic-kitchens100/epic_kitchen_videos",
        "test_video_path": "data/epic-kitchens100/epic_kitchen_videos",
        "train_label_path": "data/epic-kitchens100/EPIC_100_train.csv",
        "test_label_path": "data/epic-kitchens100/EPIC_100_validation.csv"
    }
}
