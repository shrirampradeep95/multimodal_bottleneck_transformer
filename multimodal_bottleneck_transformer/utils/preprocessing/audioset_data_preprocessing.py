# Import relevant libraries
import pandas as pd


class FormatAudioSetData:
    """
    A utility class to process and format AudioSet metadata for training or testing.
    This class reads the label mapping file and the segment files and constructs a dictionary mapping
    video segments to their corresponding human-readable labels.
    """

    def __init__(self, parameters, data_type):
        """
        Initialize the data formatter for AudioSet.

        Args:
            parameters: Dictionary containing paths to data files.
            data_type: Either 'train' or 'test', to load corresponding segment files.
        """
        self.data_type = data_type
        if self.data_type == 'train':
            self.video_path = parameters['audio_set_paths']['train_video_path']
            self.label_path = parameters['audio_set_paths']['train_label_path']
        else:
            self.video_path = parameters['audio_set_paths']['test_video_path']
            self.label_path = parameters['audio_set_paths']['test_label_path']

        self.class_label_mapping = parameters['audio_set_paths']['class_label_mapping']

        self.class_labels_df = None
        self.valid_mids = None
        self.label_dict = None
        self.all_label_df = None

    def load_class_labels(self):
        """
        Load the AudioSet class label mapping. Also sets the valid MIDs set for downstream filtering.
        """
        self.class_labels_df = pd.read_csv(self.class_label_mapping)
        self.all_label_df = self.class_labels_df.copy()
        self.valid_mids = set(self.class_labels_df['mid'])

    @staticmethod
    def _parse_segments_file(path):
        """
        Parse an AudioSet segment CSV file.
        Args:
            path: Path to the segment file.
        Returns:
            pd.DataFrame: Labels dataframe
        """
        rows = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(",", 3)
                if len(parts) == 4:
                    ytid, start, end, labels = parts
                    rows.append((
                        ytid.strip(),
                        float(start),
                        float(end),
                        labels.strip().strip('"')
                    ))
        return pd.DataFrame(rows, columns=["YTID", "start_seconds", "end_seconds", "positive_labels"])

    def _create_label_dict(self, df):
        """
        Construct a dictionary mapping video segments to display names of valid labels.
        Args:
            df: Parsed segments DataFrame.
        Returns:
            dict: Mapping from "<YTID>_<start>.000" to list of label names.
        """
        # Create a mapping from MID to display name
        mid_to_name = dict(zip(self.class_labels_df['mid'], self.class_labels_df['display_name']))

        def label_names(label_str):
            mids = [label.strip() for label in label_str.split(',') if label.strip() in self.valid_mids]
            return [mid_to_name[mid] for mid in mids]

        # Add human-readable label names to the DataFrame
        df['label_names'] = df['positive_labels'].apply(label_names)

        # Construct the dictionary
        return {
            f"{row['YTID']}_{int(row['start_seconds'])}.000": row['label_names']
            for _, row in df.iterrows()
        }

    def map_labels(self):
        """
        Main method to load label mappings and return video path and segment-to-label dictionary.

        Returns:
            tuple:
                str: Path to the video features.
                dict: Dictionary mapping "<YTID>_<start>.000" to list of label display names.
        """
        self.load_class_labels()
        label_data = self._parse_segments_file(self.label_path)
        self.label_dict = self._create_label_dict(label_data)
        return self.video_path, self.label_dict
