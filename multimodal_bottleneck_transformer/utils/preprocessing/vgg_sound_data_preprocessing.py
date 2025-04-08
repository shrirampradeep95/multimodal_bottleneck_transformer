# Import relevant libraries
import pandas as pd
import os


class FormatVGGSoundData:
    def __init__(self, parameters, data_type):
        """
        Format handler for VGGSound dataset with all videos in one folder and single-label annotations.

        Args:
            parameters (dict): Configuration dict with paths.
            data_type (str): 'train' or 'test'
        """
        self.data_type = data_type
        self.label_path = parameters['vgg_sound_paths']['label_file']
        self.video_path = parameters['vgg_sound_paths']['video_dir']
        self.label_dict = None
        self.class_labels_df = None
        self.all_label_df = None

    def _parse_vggsound_csv(self):
        """
        Parses the vggsound.csv file without headers.
        Assumes columns: YTID, start_seconds, label, train_test
        Filters rows by self.data_type.
        Constructs key as YTID_start_seconds.mp4
        """
        df = pd.read_csv(self.label_path, header=None)
        df.columns = ["YTID", "start_seconds", "label", "train_test"]

        # Filter by train or test split
        self.all_label_df = df.copy()
        self.class_labels_df = df[df["train_test"] == self.data_type].copy()
        self.class_labels_df.reset_index(drop=True, inplace=True)

        # Format start_seconds as integer with 3 digits and append .000 for filename
        self.class_labels_df["key"] = df.apply(lambda row: f"{row['YTID']}_{int(row['start_seconds'])}", axis=1)

    def _create_label_dict(self):
        """
        Create a dictionary: filename -> [label]
        """
        label_dict = {}
        for _, row in self.class_labels_df.iterrows():
            label_dict[row['key']] = [row['label']]
        return label_dict

    def map_labels(self):
        """
        Loads and maps VGGSound labels into a dict for DataLoader.
        Returns:
            video_dir (str): Path to video files
            label_dict (dict): Mapping of filename -> [label]
        """
        self._parse_vggsound_csv()
        self.label_dict = self._create_label_dict()
        return self.video_path, self.label_dict
