# AudioVisual Dataset and Loader Manager
import os
import ast
import torch
import pandas as pd
import subprocess
import tempfile
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as V
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import to_pil_image


class AudioVisualDataset(Dataset):
    """
    Custom Dataset for loading audio-visual data from video files.
    Supports AudioSet, VGGSound, and Epic-Kitchens with spectrogram and frame extraction.
    """
    def __init__(self, metadata_csv, video_root, label_to_index,
                 t_seconds=8, fps=25, dataset_type='audioset', split='train', multi_label=True):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.video_root = video_root
        self.label_to_index = label_to_index
        self.t_seconds = t_seconds
        self.fps = fps
        self.split = split
        self.dataset_type = dataset_type
        self.multi_label = multi_label
        self.sample_rate = 16000

        self.video_transform = self._build_video_transform()
        self.audio_transform = self._build_audio_transform()

    def _build_video_transform(self):
        if self.split == 'train':
            return V.Compose([
                V.Resize((256, 256)),
                V.RandomCrop((224, 224)),
                V.RandomHorizontalFlip(),
                V.ColorJitter(0.2, 0.2, 0.2, 0.1),
                V.ToTensor()
            ])
        else:
            return V.Compose([
                V.Resize((224, 224)),
                V.ToTensor()
            ])

    def _build_audio_transform(self):
        base = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(0.025 * self.sample_rate),
            hop_length=int(0.010 * self.sample_rate),
            n_mels=128
        )
        if self.split == 'train':
            return torch.nn.Sequential(
                base,
                T.FrequencyMasking(freq_mask_param=48),
                T.TimeMasking(time_mask_param=192)
            )
        return base

    def _load_video(self, video_path):
        """
        Extract and sample frames from video using OpenCV.
        Handles Epic-Kitchens and other datasets with dataset-specific sampling strategy.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25

        if self.dataset_type.lower() == "epic":
            self.num_frames = 32
            self.stride = 1
        else:
            self.num_frames = 8
            self.stride = int((self.t_seconds * video_fps) / self.num_frames)

        required = self.num_frames * self.stride
        if total_frames < required:
            cap.release()
            raise RuntimeError(f"Not enough frames in video: {video_path}")

        frame_indices = [0 + i * self.stride for i in range(self.num_frames)]
        frames, current_frame = [], 0

        success = True
        while success and len(frames) < self.num_frames:
            success, frame = cap.read()
            if not success:
                break
            if current_frame in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = to_pil_image(frame_rgb)
                frames.append(self.video_transform(pil_img))
            current_frame += 1

        cap.release()

        if len(frames) != self.num_frames:
            raise RuntimeError(f"Could not extract {self.num_frames} frames from {video_path}")

        return torch.stack(frames)  # [T, 3, 224, 224]

    def _load_audio(self, video_file_path):
        """
        Extract mono-channel audio using ffmpeg, pad/trim to fixed length,
        and convert to log-mel spectrogram.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            command = ["ffmpeg", "-i", video_file_path, "-ac", "1", "-ar", str(self.sample_rate), "-vn", "-y", tmp_wav.name]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            waveform, sr = torchaudio.load(tmp_wav.name)

        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        desired_len = self.sample_rate * self.t_seconds
        waveform = torch.nn.functional.pad(waveform, (0, max(0, desired_len - waveform.size(1))))
        waveform = waveform[:, :desired_len]
        mel_spec = self.audio_transform(waveform)
        return mel_spec  # [1, 128, T]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_name = row["file_name"]
        label_data = row["mapped_labels"] if self.multi_label else row["mapped_label"]

        video_path = os.path.join(self.video_root, f"{file_name}.mp4")
        video = self._load_video(video_path)
        audio = self._load_audio(video_path)

        if self.multi_label:
            labels = ast.literal_eval(label_data)
            label_tensor = torch.zeros(len(self.label_to_index))
            for lbl in labels:
                if lbl in self.label_to_index:
                    label_tensor[self.label_to_index[lbl]] = 1.0
        else:
            label_tensor = torch.tensor(self.label_to_index[label_data], dtype=torch.long)

        return {"video": video, "audio": audio, "labels": label_tensor}

    def __len__(self):
        return len(self.metadata)


class AVSubsetDataManager:
    """
    DataLoader manager class that supports top-K label filtering and multi-label/single-label modes.
    """
    def __init__(self, metadata_csv, video_root, batch_size=8, num_workers=4,
                 dataset_type="audioset", top_k_labels=None):
        self.metadata_csv = metadata_csv
        self.video_root = video_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        self.top_k_labels = top_k_labels
        self.multi_label = dataset_type == 'AudioSet'

        self.train_loader = None
        self.test_loader = None
        self.label_to_index = None

    @staticmethod
    def av_collate_fn(batch):
        videos = [x["video"] for x in batch]
        audios = [x["audio"] for x in batch]
        labels = [x["labels"] for x in batch]

        videos = torch.stack(videos)
        audios = pad_sequence([a.squeeze(0).transpose(0, 1) for a in audios], batch_first=True)
        audios = audios.transpose(1, 2).unsqueeze(1)
        labels = torch.stack(labels)

        return videos, audios, labels

    def _get_top_k_label_set(self, df, label_col):
        if self.multi_label:
            all_labels = []
            for row in df[label_col]:
                all_labels.extend(ast.literal_eval(row))
            label_counts = pd.Series(all_labels).value_counts()
        else:
            label_counts = df[label_col].value_counts()
        return set(label_counts.head(self.top_k_labels).index)

    def _filter_and_map_labels(self, df, selected_labels, label_col):
        if self.multi_label:
            filtered = df[df[label_col].apply(lambda l: any(lbl in selected_labels for lbl in ast.literal_eval(l)))].copy()
            filtered[label_col] = filtered[label_col].apply(
                lambda l: str([lbl for lbl in ast.literal_eval(l) if lbl in selected_labels])
            )
        else:
            filtered = df[df[label_col].isin(selected_labels)].copy()
        return filtered

    def load_format_data(self):
        df = pd.read_csv(self.metadata_csv)
        label_col = "mapped_labels" if self.multi_label else "mapped_label"

        if self.top_k_labels:
            selected_labels = self._get_top_k_label_set(df, label_col)
            df = self._filter_and_map_labels(df, selected_labels, label_col)
            self.label_to_index = {lbl: i for i, lbl in enumerate(sorted(selected_labels))}
        else:
            if self.multi_label:
                label_set = set(lbl for row in df[label_col] for lbl in ast.literal_eval(row))
            else:
                label_set = set(df[label_col].unique())
            self.label_to_index = {lbl: i for i, lbl in enumerate(sorted(label_set))}

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        train_df.to_csv("_tmp_train.csv", index=False)
        test_df.to_csv("_tmp_test.csv", index=False)

        self.train_dataset = AudioVisualDataset(
            metadata_csv="_tmp_train.csv",
            video_root=self.video_root,
            label_to_index=self.label_to_index,
            dataset_type=self.dataset_type,
            split="train",
            multi_label=self.multi_label
        )

        self.test_dataset = AudioVisualDataset(
            metadata_csv="_tmp_test.csv",
            video_root=self.video_root,
            label_to_index=self.label_to_index,
            dataset_type=self.dataset_type,
            split="test",
            multi_label=self.multi_label
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=self.num_workers,
                                       collate_fn=self.av_collate_fn)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.num_workers,
                                      collate_fn=self.av_collate_fn)

    def get_data_loaders(self):
        if self.train_loader is None or self.test_loader is None:
            self.load_format_data()
        return self.train_loader, self.test_loader
