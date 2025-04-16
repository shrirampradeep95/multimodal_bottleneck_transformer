import os
import pandas as pd

def create_combined_vggsound_csv(label_file_path, video_dir, output_csv="vggsound_combined_metadata.csv"):
    """
    Creates a combined CSV for VGGSound with the following columns:
    - file_name (YTID_start)
    - video_file_name (e.g., <file_name>.mp4)
    - mapped_label
    - split (train/test)
    Only includes entries for which video files exist in the video_dir.

    Args:
        label_file_path (str): Path to the VGGSound CSV file (no header).
        video_dir (str): Directory where .mp4 video files are stored.
        output_csv (str): Path to save the filtered combined CSV.
    """
    # Load the label CSV
    df = pd.read_csv(label_file_path, header=None)
    df.columns = ["YTID", "start_seconds", "label", "split"]

    # Construct file_name as "<YTID>_<start_seconds>"
    df["file_name"] = df.apply(lambda row: f"{row['YTID']}_{int(row['start_seconds'])}", axis=1)
    df["video_file_name"] = df["file_name"] + ".mp4"

    # Get list of available .mp4 files
    available_files = set(os.listdir(video_dir))

    # Filter to only rows that exist in the video directory
    df = df[df["video_file_name"].isin(available_files)]

    # Rename and select columns
    df.rename(columns={"label": "mapped_label"}, inplace=True)
    output_df = df[["file_name", "video_file_name", "mapped_label", "split"]]

    # Save to output
    output_df.to_csv(output_csv, index=False)
    print(f"[INFO] Filtered VGGSound metadata saved to: {output_csv}")
    print(f"[INFO] Rows retained: {len(output_df)}")


if __name__ == "__main__":
    create_combined_vggsound_csv(
        label_file_path="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/vggsound/vggsound.csv",
        video_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/vggsound/vggsound_videos",
        output_csv="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/vggsound/vggsound_combined_metadata.csv"
    )
