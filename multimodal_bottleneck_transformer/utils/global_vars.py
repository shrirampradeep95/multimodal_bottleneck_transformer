# Import relevant libraries
import torch

parameters = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "location":
        "/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/",
    "dataset": "AudioSet",
    "audio_set_paths": {
        "train_video_path": "data/audioset/balanced_train_segments",
        "test_video_path": "data/audioset/eval_segments",
        "train_label_path": "data/audioset/balanced_train_segments.csv",
        "test_label_path": "data/audioset/eval_segments.csv",
        "class_label_mapping": "data/audioset/class_labels_indices.csv"
    },
    "vgg_sound_paths": {
        "video_dir": "data/vggsound/vggsound_videos",
        "label_file": "data/vggsound/vggsound.csv",
    },
    "epic_kitchens_paths": {
        "train_video_path": "data/epic-kitchens100/epic_kitchen_videos",
        "test_video_path": "data/epic-kitchens100/epic_kitchen_videos",
        "train_label_path": "data/epic-kitchens100/EPIC_100_train.csv",
        "test_label_path": "data/epic-kitchens100/EPIC_100_validation.csv"
    },
    "batch_size": 2,
    "lr": 0.5,
    "epochs": 50,
    "mixup_alpha": 0.3,
    "t_seconds": 8,
    "save_every": 10,
    "num_workers": 4,
    "videos_to_use": None
}
