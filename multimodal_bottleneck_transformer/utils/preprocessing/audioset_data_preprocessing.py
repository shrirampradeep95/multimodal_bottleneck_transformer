# Import relevant libraries
import pandas as pd


class FormatAudioSetData:
    def __init__(self, parameters, data_type):
        """
        Initialize class parameters
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
        Load the AudioSet class label mapping (mid -> display_name)
        """
        self.class_labels_df = pd.read_csv(self.class_label_mapping)
        self.all_label_df = self.class_labels_df.copy()
        self.valid_mids = set(self.class_labels_df['mid'])

    @staticmethod
    def _parse_segments_file(path):
        """
        Parse an AudioSet segments file robustly and return a DataFrame
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

    def _map_labels_to_mids(self, label_str):
        """
        Convert comma-separated string of mids to a list of valid mids
        """
        return [label.strip() for label in label_str.split(',') if label.strip() in self.valid_mids]

    def _create_label_dict(self, df):
        """
        Create label dictionary from a DataFrame of AudioSet segments.
        Dictionary values contain only display names of the labels.
        """
        mid_to_name = dict(zip(self.class_labels_df['mid'], self.class_labels_df['display_name']))

        def label_names(label_str):
            mids = [mid.strip() for mid in label_str.split(',') if mid.strip() in self.valid_mids]
            return [mid_to_name[mid] for mid in mids]

        df['label_names'] = df['positive_labels'].apply(label_names)

        return {
            f"{row['YTID']}_ {int(row['start_seconds'])}.000": row['label_names']
            for _, row in df.iterrows()
        }

    def map_labels(self):
        """
        Map training and testing labels to dictionaries
        """
        self.load_class_labels()

        # Parse segment files
        if self.data_type == 'train':
            label_data = self._parse_segments_file(self.label_path)

            # Create label dictionaries
            self.label_dict = self._create_label_dict(label_data)
        else:
            label_data = self._parse_segments_file(self.label_path)

            # Create label dictionaries
            self.label_dict = self._create_label_dict(label_data)

        return self.video_path, self.label_dict

