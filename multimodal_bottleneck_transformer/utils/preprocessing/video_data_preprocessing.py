# Import relevant libraries
import torch
import torchaudio
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import os
import glob
import subprocess
import tempfile

# Import user defined libraries
from utils.preprocessing.audioset_data_preprocessing import FormatAudioSetData
from utils.preprocessing.vgg_sound_data_preprocessing import FormatVGGSoundData


class VideoDataset(Dataset):
    def __init__(self, parameters, videos_to_use, data_type, t_seconds=8):
        """
        Dataset to load video and audio data for AudioSet-based training.

        Args:
            parameters (dict): Configuration dictionary
            data_type (str): 'train' or 'test'
            t_seconds (int): Length of audio/video clips to extract
            videos_to_use (int or None): If set, limits number of videos loaded
        """
        if parameters['dataset'] == 'AudioSet':
            formatter = FormatAudioSetData(parameters, data_type)
            self.video_dir, self.label_dict = formatter.map_labels()
            sorted_df = formatter.class_labels_df.sort_values(by='index')
            self.label_to_index = dict(zip(sorted_df['display_name'], sorted_df['index']))
        elif parameters['dataset'] == 'vggsound':
            formatter = FormatVGGSoundData(parameters, data_type)
            self.video_dir, self.label_dict = formatter.map_labels()
            unique_labels = sorted(formatter.all_label_df["label"].unique())
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}\

        self.video_paths = sorted(glob.glob(os.path.join(self.video_dir, "*.mp4")))

        # Truncate the dataset if specified
        if videos_to_use:
            self.video_paths = self.video_paths[:videos_to_use]

        # Store YTID_start.000 labels per video file
        self.video_keys = [self._extract_key_from_path(p) for p in self.video_paths]
        self.labels = [self.label_dict.get(k, []) for k in self.video_keys]

        self.t_seconds = t_seconds
        self.fps = 25
        self.num_frames = 8
        self.stride = int((t_seconds * self.fps) / self.num_frames)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=128
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    @staticmethod
    def _extract_key_from_path(path):
        """
        Extract YTID_start.000 format from file path.
        Example: /path/to/--PJHxphWEs_030.000.mp4 → --PJHxphWEs_030.000
        """
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]

    def __len__(self):
        return len(self.video_paths)

    def _sample_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * self.stride, total_frames - 1))
            success, frame = cap.read()
            if not success or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame).resize((224, 224))
            frame = torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255
            frames.append(frame)
        cap.release()
        return torch.stack(frames)

    def _extract_audio(self, path):
        """
        Extracts audio using ffmpeg into a temporary .wav file,
        loads it with torchaudio, and returns log-mel spectrogram.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                # Extract audio using ffmpeg
                command = [
                    "ffmpeg",
                    "-i", path,
                    "-ar", "16000",  # Resample to 16kHz
                    "-ac", "1",  # Convert to mono
                    "-y",  # Overwrite if exists
                    tmp_wav.name
                ]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                # Load with torchaudio
                wav, sr = torchaudio.load(tmp_wav.name)

            wav = wav[:, :self.t_seconds * 16000]
            mel = self.mel(wav)
            logmel = self.db(mel)

            # Make sure output is (1, 128, 800)
            logmel = logmel[:, :, :800]
            if logmel.shape[-1] < 800:
                logmel = torch.nn.functional.pad(logmel, (0, 800 - logmel.shape[-1]))

            return logmel

        except Exception as e:
            print(f"[Audio Extraction Failed] {path}: {e}")
            return torch.zeros((1, 128, 800))

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        video = self._sample_frames(path)
        audio = self._extract_audio(path)

        label_names = self.labels[idx]
        label_tensor = torch.zeros(len(self.label_to_index))
        for name in label_names:
            if name in self.label_to_index:
                label_tensor[self.label_to_index[name]] = 1.0

        return video, audio, label_tensor
