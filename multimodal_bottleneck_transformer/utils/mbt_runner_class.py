# Import relevant libraries
from torch.utils.data import DataLoader

# Import user defined libraries
from utils.preprocessing.video_data_preprocessing import VideoDataset
from utils.train_eval.trainer_evaluator import TrainerEvaluator
from utils.model.multimodal_transformer import MultimodalTransformer


class MBTRunnerClass:
    """
    Runner class to manage training of the multimodal transformer with bottleneck fusion.
    """
    def __init__(self, parameters):
        """
        Intialize the required variables
        :param parameters: Model parameters dictionary
        """
        self.parameters = parameters.copy()
        self.device = self.parameters['device']
        self.modality = self.parameters.get("modality", "av")

    def load_format_data(self):
        """
        Function to load and format audio and video
        """
        # 1. Get number of classes
        # 2. Transform train data and test data
        # Load dataset
        train_dataset = VideoDataset(
            self.parameters, videos_to_use=self.parameters["videos_to_use"], data_type='train'
        )
        val_dataset = VideoDataset(
            self.parameters, videos_to_use=self.parameters["videos_to_use"], data_type='test'
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.parameters["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.parameters["batch_size"], shuffle=True, num_workers=4
        )

        model = MultimodalTransformer(num_classes=len(train_dataset.label_to_index), modality=self.modality)

        trainer = TrainerEvaluator(
            self.parameters,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_to_index=train_dataset.label_to_index,
            dataset=self.parameters['dataset'],
            device=self.device
        )
        trainer.train(epochs=self.parameters.get("epochs", 10))
