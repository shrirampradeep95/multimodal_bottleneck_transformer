import os
import cv2
import torchaudio
import subprocess
import tempfile
import pandas as pd
import shutil


def is_valid_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        return success and frame is not None
    except Exception:
        return False


def is_valid_audio(video_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-ar", "16000", "-ac", "1",
                "-y", tmp_wav.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            wav, sr = torchaudio.load(tmp_wav.name)
        return wav.shape[-1] > 0
    except Exception:
        return False


def validate_and_move_bad_files(video_dir, metadata_csv_path, bad_videos_dir, cleaned_metadata_csv_path=None):
    # Load metadata
    df = pd.read_csv(metadata_csv_path)

    if not os.path.exists(bad_videos_dir):
        os.makedirs(bad_videos_dir)

    # Track valid files
    valid_file_names = []

    print(f"[INFO] Validating {len(df)} video files...")

    for idx, row in df.iterrows():
        file_name = row["video_file_name"]
        video_path = os.path.join(video_dir, file_name)

        if not os.path.isfile(video_path):
            print(f"[MISSING] File not found: {file_name}")
            continue

        is_bad = False
        if not is_valid_video(video_path):
            print(f"[BAD VIDEO] {file_name}")
            is_bad = True
        elif not is_valid_audio(video_path):
            print(f"[BAD AUDIO] {file_name}")
            is_bad = True

        if is_bad:
            try:
                shutil.move(video_path, os.path.join(bad_videos_dir, file_name))
                print(f"[MOVED] {file_name} -> {bad_videos_dir}")
            except Exception as e:
                print(f"[ERROR moving {file_name}]: {e}")
        else:
            valid_file_names.append(file_name)

    # Filter metadata
    cleaned_df = df[df["video_file_name"].isin(valid_file_names)]

    if cleaned_metadata_csv_path is None:
        cleaned_metadata_csv_path = metadata_csv_path

    cleaned_df.to_csv(cleaned_metadata_csv_path, index=False)
    print(f"[INFO] Cleaned metadata saved to: {cleaned_metadata_csv_path}")
    print(f"[INFO] Total valid files retained: {len(cleaned_df)}")


if __name__ == "__main__":
    # validate_and_move_bad_files(
    #     video_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/audioset_videos",
    #     metadata_csv_path="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/audioset_combined_metadata.csv",
    #     bad_videos_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/bad_videos"
    # )
    validate_and_move_bad_files(
        video_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/vggsound/vggsound_videos",
        metadata_csv_path="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/vggsound/vggsound_combined_metadata.csv",
        bad_videos_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/vggsound/bad_videos"
    )