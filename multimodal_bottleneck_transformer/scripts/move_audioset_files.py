import os
import shutil


def move_and_rename_audioset_videos(src_dirs, dest_dir):
    """
    Move and rename AudioSet videos from multiple source folders into a single destination.

    Args:
        src_dirs (list): List of source directories containing videos.
        dest_dir (str): Destination directory to move all videos to.
    """
    os.makedirs(dest_dir, exist_ok=True)
    moved_count = 0

    for src_dir in src_dirs:
        for file_name in os.listdir(src_dir):
            if not file_name.endswith(".mp4"):
                continue

            # Check if file has the space pattern we want to fix
            if "_ " in file_name:
                new_file_name = file_name.replace("_ ", "_").replace(".000.mp4", ".mp4")
            else:
                new_file_name = file_name.replace(".000.mp4", ".mp4")

            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, new_file_name)

            if not os.path.exists(dest_path):
                shutil.move(src_path, dest_path)
                moved_count += 1
            else:
                print(f"[SKIP] Already exists: {dest_path}")

    print(f"[INFO] Total files moved and renamed: {moved_count}")


if __name__ == "__main__":
    move_and_rename_audioset_videos(
        src_dirs=[
            "/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/balanced_train_segments",
            "/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/eval_segments"
        ],
        dest_dir="/Users/shrirampradeep/Documents/Mtech - IIT H/Sem 4 - CS6880 Multimedia Content Analysis/Implementation/mulimodal_bottleneck_transformer/multimodal_bottleneck_transformer/data/audioset/audioset_videos"
    )
