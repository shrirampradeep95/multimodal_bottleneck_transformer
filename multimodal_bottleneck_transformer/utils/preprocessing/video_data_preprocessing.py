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

# Import user-defined preprocessing modules
from utils.preprocessing.audioset_data_preprocessing import FormatAudioSetData
from utils.preprocessing.vgg_sound_data_preprocessing import FormatVGGSoundData


class VideoDataset(Dataset):
    """
    A dataset class to load video and audio data for AudioSet or VGGSound datasets.
    Returns:
        - RGB frames
        - Log-mel spectrogram of audio
        - Multi-label or single-label vector
    """
    def __init__(self, parameters, videos_to_use, t_seconds=1, fps=25, num_frames=8, data_type='train'):
        """
        Initialize the dataset.
        Args:
            parameters: Configuration dictionary with paths and dataset type.
            videos_to_use: Number of videos to load
            data_type: Either 'train' or 'test'.
            t_seconds: Length of the video/audio clip in seconds.
        """
        self.data_type = data_type
        self.t_seconds = t_seconds
        self.fps = fps
        self.num_frames = num_frames
        self.stride = int((t_seconds * self.fps) / self.num_frames)

        # Load labels and paths based on dataset
        if parameters['dataset'] == 'AudioSet':
            formatter = FormatAudioSetData(parameters, data_type)
            self.video_dir, self.label_dict = formatter.map_labels()
            sorted_df = formatter.class_labels_df.sort_values(by='index')
            self.label_to_index = dict(zip(sorted_df['display_name'], sorted_df['index']))
        elif parameters['dataset'] == 'vggsound':
            formatter = FormatVGGSoundData(parameters, data_type)
            self.video_dir, self.label_dict = formatter.map_labels()
            unique_labels = sorted(formatter.all_label_df["label"].unique())
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            raise ValueError("Unsupported dataset type")

        # Collect all video file paths
        self.video_paths = sorted(glob.glob(os.path.join(self.video_dir, "*.mp4")))

        # Limit the number of videos if specified
        if videos_to_use:
            self.video_paths = self.video_paths[:videos_to_use]

        # Generate keys from video filenames and map to labels
        self.video_keys = [self._extract_key_from_path(p) for p in self.video_paths]
        self.labels = [self.label_dict.get(k, []) for k in self.video_keys]

        # Define audio transformations
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=128
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    @staticmethod
    def _extract_key_from_path(path):
        """
        Extract key from filename in the format <YTID>_<start>.000.mp4 → <YTID>_<start>.000
        Args:
            path: Path to video file
        Returns:
            str: Key used to lookup labels
        """
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]

    def __len__(self):
        return len(self.video_paths)

    def _sample_frames(self, path):
        """
        Extracts evenly spaced frames from a video file.
        Args:
            path: Path to the .mp4 file
        Returns:
            Tensor: Normalized RGB frames
        """
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
            frame = torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()
        return torch.stack(frames)

    def _extract_audio(self, path):
        """
        Extracts audio from the video file using ffmpeg and returns a log-mel spectrogram.
        Args:
            path: Path to .mp4 file
        Returns:
            Tensor: Log-mel spectrogram
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                command = [
                    "ffmpeg", "-i", path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    tmp_wav.name
                ]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                wav, sr = torchaudio.load(tmp_wav.name)

            # Truncate or pad to t_seconds
            wav = wav[:, :self.t_seconds * 16000]
            mel = self.mel(wav)
            logmel = self.db(mel)

            # Ensure (1, 128, 800) shape
            logmel = logmel[:, :, :800]
            if logmel.shape[-1] < 800:
                logmel = torch.nn.functional.pad(logmel, (0, 800 - logmel.shape[-1]))

            return logmel

        except Exception as e:
            print(f"[Audio Extraction Failed] {path}: {e}")
            return torch.zeros((1, 128, 800))

    def __getitem__(self, idx):
        """
        Retrieves the (video, audio, label) tuple for the given index.
        Args:
            idx: Index in dataset
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                video, audio, label vector
        """
        path = self.video_paths[idx]
        video = self._sample_frames(path)
        audio = self._extract_audio(path)

        label_names = self.labels[idx]
        label_tensor = torch.zeros(len(self.label_to_index))
        for name in label_names:
            if name in self.label_to_index:
                label_tensor[self.label_to_index[name]] = 1.0

        return video, audio, label_tensor
