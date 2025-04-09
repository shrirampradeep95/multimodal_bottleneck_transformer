# Import relevant libraries
from torch.utils.data import DataLoader

# Import user-defined libraries
from utils.preprocessing.video_data_preprocessing import VideoDataset
from utils.train_eval.trainer_evaluator import TrainerEvaluator
from utils.model.multimodal_transformer import AVModel, AudioOnlyModel, VideoOnlyModel


class MBTRunnerClass:
    """
    Runner class to manage the full training lifecycle of the Multimodal Bottleneck Transformer (MBT)
    model for audiovisual classification.
    """
    def __init__(self, parameters):
        """
        Initialize the runner with parameters.
        Args:
            parameters: Configuration dictionary containing model, data, and training parameters.
        """
        self.parameters = parameters.copy()
        self.device = self.parameters['device']
        self.modality = self.parameters.get("modality", "av")
        print("[INFO] MBT Runner initialized with modality:", self.modality)

    def load_format_data(self):
        """
        Load and prepare audio-visual training and validation datasets, create data loaders, initialize
        the model and trainer, and start training.
        """
        print("[INFO] Preparing datasets")
        # Load training dataset
        train_dataset = VideoDataset(
            self.parameters,
            videos_to_use=self.parameters["videos_to_use"],
            t_seconds=self.parameters['t_seconds'],
            fps=self.parameters['fps'],
            num_frames=self.parameters['num_frames'],
            data_type='train'
        )
        print(f"[INFO] Loaded training dataset with {len(train_dataset)} samples.")

        # Load validation dataset
        val_dataset = VideoDataset(
            self.parameters,
            videos_to_use=self.parameters["videos_to_use"],
            t_seconds=self.parameters['t_seconds'],
            fps=self.parameters['fps'],
            num_frames=self.parameters['num_frames'],
            data_type='test',
        )
        print(f"[INFO] Loaded validation dataset with {len(val_dataset)} samples.")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=self.parameters['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=self.parameters['num_workers']
        )
        print("[INFO] DataLoaders initialized.")

        if self.parameters['modality'] == 'av':
            # Instantiate the AVModel with appropriate number of classes and bottlenecks
            model = AVModel(
                num_classes=len(train_dataset.label_to_index),
                num_bottlenecks=self.parameters['num_bottlenecks'],
                transformer_layers=self.parameters['transformer_layers']
            )
            print("[INFO] AVModel instantiated with", len(train_dataset.label_to_index), "classes.")
        elif self.parameters['modality'] == 'a':
            # Instantiate the AudioOnlyModel with appropriate number of classes
            model = AudioOnlyModel(
                num_classes=len(train_dataset.label_to_index)
            )
            print("[INFO] Audio model instantiated with", len(train_dataset.label_to_index), "classes.")
        else:
            # Instantiate the VideoOnlyModel with appropriate number of classes
            model = VideoOnlyModel(
                num_classes=len(train_dataset.label_to_index)
            )
            print("[INFO] Video Model instantiated with", len(train_dataset.label_to_index), "classes.")

        # Create the trainer/evaluator
        trainer = TrainerEvaluator(
            parameters=self.parameters,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_to_index=train_dataset.label_to_index,
            dataset=self.parameters['dataset'],
            device=self.device,
            vgg_sound_lr=self.parameters['vgg_sound_lr'],
            audio_set_lr=self.parameters['audio_set_lr']
        )
        print("[INFO] TrainerEvaluator initialized.")

        # Start training
        print("[INFO] Starting training for", self.parameters.get("epochs", 10), "epochs...\n")
        trainer.train(epochs=self.parameters.get("epochs", 10))
        print("[INFO] Training complete.")
