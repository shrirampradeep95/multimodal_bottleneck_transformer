# Import user-defined libraries
from utils.preprocessing.load_data import AVSubsetDataManager
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
        # Initialise data manager
        manager = AVSubsetDataManager(
            metadata_csv=self.parameters[self.parameters['dataset']]['metadata_file'],
            video_root=self.parameters[self.parameters['dataset']]['video_dir'],
            batch_size=self.parameters['batch_size'],
            num_workers=self.parameters['num_workers'],
            dataset_type=self.parameters['dataset'],
            top_k_labels=self.parameters['top_k_labels']
        )

        manager.load_format_data()
        train_loader, test_loader = manager.get_data_loaders()

        print("[INFO] DataLoaders initialized.")

        if self.parameters['modality'] == 'av':
            # Instantiate the AVModel with appropriate number of classes and bottlenecks
            model = AVModel(
                num_classes=len(manager.label_to_index),
                num_bottlenecks=self.parameters['num_bottlenecks'],
                transformer_layers=self.parameters['transformer_layers']
            )
            print("[INFO] AVModel instantiated with", len(manager.label_to_index), "classes.")
        elif self.parameters['modality'] == 'a':
            # Instantiate the AudioOnlyModel with appropriate number of classes
            model = AudioOnlyModel(
                num_classes=len(manager.label_to_index)
            )
            print("[INFO] Audio model instantiated with", len(manager.label_to_index), "classes.")
        else:
            # Instantiate the VideoOnlyModel with appropriate number of classes
            model = VideoOnlyModel(
                num_classes=len(manager.label_to_index)
            )
            print("[INFO] Video Model instantiated with", len(manager.label_to_index), "classes.")

        # Create the trainer/evaluator
        trainer = TrainerEvaluator(
            parameters=self.parameters,
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            label_to_index=manager.label_to_index,
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
