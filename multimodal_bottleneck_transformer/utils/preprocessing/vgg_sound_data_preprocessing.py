# Import relevant libraries
import pandas as pd


class FormatVGGSoundData:
    """
    Formatter class for the VGGSound dataset, where all video clips are in one directory
    and annotations contain a single label per clip.
    """
    def __init__(self, parameters, data_type):
        """
        Initialize formatter for VGGSound dataset.

        Args:
            parameters: Configuration dictionary
            data_type: Either 'train' or 'test' to determine split.
        """
        self.data_type = data_type
        self.label_path = parameters['vgg_sound_paths']['label_file']
        self.video_path = parameters['vgg_sound_paths']['video_dir']

        self.label_dict = None
        self.class_labels_df = None
        self.all_label_df = None

    def _parse_vggsound_csv(self):
        """
        Parses the VGGSound CSV file. The CSV contains no headers by default. This method filters the CSV
        based on the data_type ('train' or 'test') and constructs a new 'key' column used for mapping to filenames.
        """
        df = pd.read_csv(self.label_path, header=None)
        df.columns = ["YTID", "start_seconds", "label", "train_test"]

        self.all_label_df = df.copy()

        # Filter DataFrame for current split
        self.class_labels_df = df[df["train_test"] == self.data_type].copy()
        self.class_labels_df.reset_index(drop=True, inplace=True)

        # Create a filename key: "<YTID>_<start_seconds>"
        self.class_labels_df["key"] = self.class_labels_df.apply(
            lambda row: f"{row['YTID']}_{int(row['start_seconds'])}", axis=1
        )

    def _create_label_dict(self):
        """
        Create a label dictionary where keys are "<YTID>_<start_seconds>" and values are single-label lists.
        Returns:
            dict: {"video_filename": [label]}
        """
        label_dict = {}
        for _, row in self.class_labels_df.iterrows():
            label_dict[row['key']] = [row['label']]
        return label_dict

    def map_labels(self):
        """
        Parses the VGGSound label CSV and returns a mapping from video filenames to labels.
        Returns:
            str: Path to the directory containing all video files.
            dict: Mapping from filename key to a list with a single label.
        """
        self._parse_vggsound_csv()
        self.label_dict = self._create_label_dict()
        return self.video_path, self.label_dict
