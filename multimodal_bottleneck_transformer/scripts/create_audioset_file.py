import os
import pandas as pd


def parse_segments_file(segment_path, split):
    """
    Parses an AudioSet segments file and attaches the split label.
    Converts video name from {ytid}_ {start}.000.mp4 → {ytid}_{start}.mp4
    """
    rows = []
    with open(segment_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",", 3)
            if len(parts) == 4:
                ytid, start, end, labels = parts
                ytid = ytid.strip()
                start_time = int(float(start))
                key = f"{ytid}_{start_time}"  # Cleaned key
                label_list = labels.strip().strip('"').split(",")
                rows.append({
                    "file_name": key,
                    "video_file_name": f"{key}.mp4",
                    "mids": [mid.strip() for mid in label_list],
                    "split": split
                })
    return pd.DataFrame(rows)


def load_label_mapping(label_path):
    """
    Loads class_labels_indices.csv and returns MID → display_name dict.
    """
    df = pd.read_csv(label_path)
    return dict(zip(df["mid"], df["display_name"]))


def create_combined_audioset_csv(
    train_segments_path,
    test_segments_path,
    label_map_path,
    video_dir,
    output_csv="audioset_combined_metadata.csv"
):
    """
    Creates a filtered combined AudioSet metadata CSV with:
    - file_name (ytid_start)
    - video_file_name (ytid_start.mp4)
    - mapped_labels
    - split
    Only includes entries where the video file exists in the video_dir.
    """
    # Load MID → Label mapping
    label_map = load_label_mapping(label_map_path)

    # Load segment metadata
    train_df = parse_segments_file(train_segments_path, "train")
    test_df = parse_segments_file(test_segments_path, "test")

    # Combine metadata
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Get available video filenames
    available_files = set(os.listdir(video_dir))

    # Filter only those with actual video files
    combined_df = combined_df[combined_df["video_file_name"].isin(available_files)]

    # Map MIDs to display names
    combined_df["mapped_labels"] = combined_df["mids"].apply(
        lambda mids: [label_map[mid] for mid in mids if mid in label_map]
    )

    # Final output
    output_df = combined_df[["file_name", "video_file_name", "mapped_labels", "split"]]
    output_df.to_csv(output_csv, index=False)

    print(f"[INFO] Filtered AudioSet metadata saved to: {output_csv}")
    print(f"[INFO] Total entries: {len(output_df)}")


if __name__ == "__main__":
    create_combined_audioset_csv(
        train_segments_path="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/balanced_train_segments.csv",
        test_segments_path="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/eval_segments.csv",
        label_map_path="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/class_labels_indices.csv",
        video_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/audioset_videos",
        output_csv="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/audioset_combined_metadata.csv"
    )
